import os
import random
import torch

from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids, filter_invalid_examples
from utils import get_input_files
from data_utils import to_positive_negative_format
from tasks import get_possible_answers_by_task_name


def _filter_out_same_label_negatives(corpus: Dataset, example: Dict) -> Dict:
    # For text classification tasks, we need to filter out negatives examples with the same label
    # multiple choice
    if len(example['options']) > 1:
        return example
    # not text classification task
    task_name = example['task_name']
    if not get_possible_answers_by_task_name(task_name):
        return example

    answer = example['answers'][0].strip()
    new_doc_ids: List[str] = []
    new_doc_scores: List[float] = []
    for idx, doc_id in enumerate(example['negatives']['doc_id']):
        cur_ex: dict = corpus[int(doc_id)]
        contents: str = cur_ex['contents']
        input_ans: str = contents.strip().split('\n')[-1]
        if cur_ex['task_name'] != task_name or input_ans != answer:
            new_doc_ids.append(doc_id)
            new_doc_scores.append(example['negatives']['score'][idx])
    example['negatives']['doc_id'] = new_doc_ids
    example['negatives']['score'] = new_doc_scores
    return example


class BiencoderDataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments,
                 input_files: List[str],
                 tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.input_files = input_files
        self.tokenizer = tokenizer
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0

        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train', keep_in_memory=True)

        self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')
        with self.args.main_process_first(desc="pre-processing"):
            self.dataset = filter_invalid_examples(args, self.dataset)
            self.dataset = self.dataset.map(
                partial(to_positive_negative_format,
                        topk_as_positive=args.topk_as_positive,
                        bottomk_as_negative=args.bottomk_as_negative),
                load_from_cache_file=args.world_size > 1,
                desc='to_positive_negative_format',
                remove_columns=['doc_ids', 'doc_scores']
            )
            self.dataset = self.dataset.map(
                partial(_filter_out_same_label_negatives, self.corpus),
                load_from_cache_file=args.world_size > 1,
                desc='filter_out_same_label_negatives',
            )

        if self.args.max_train_samples is not None:
            self.dataset = self.dataset.select(range(self.args.max_train_samples))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.dataset)), 1):
            logger.info(f"Sample {index} of the training set: {self.dataset[index]}.")

        self.dataset.set_transform(self._transform_func)

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        examples = deepcopy(examples)
        # add some random negatives if not enough
        for idx in range(len(examples['query_id'])):
            while len(examples['negatives'][idx]['doc_id']) < self.negative_size:
                random_doc_id = str(random.randint(0, len(self.corpus) - 1))
                examples['negatives'][idx]['doc_id'].append(random_doc_id)
                examples['negatives'][idx]['score'].append(-100.)

        epoch = int(self.trainer.state.epoch or 0) if self.trainer is not None else 0
        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=epoch + self.args.seed
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        queries = examples['query'][:]
        input_docs: List[str] = [self.corpus[doc_id]['contents'] for doc_id in input_doc_ids]

        if self.args.add_qd_prompt:
            queries = ['query: ' + q for q in queries]
            input_docs = ['query: ' + d for d in input_docs]

        step_size = self.args.train_n_passages
        batch_dict = {
            'query': queries,
            'passages': [input_docs[idx:idx + step_size] for idx in range(0, len(input_docs), step_size)],
        }
        assert len(batch_dict['query']) == len(batch_dict['passages']), f'{len(batch_dict["query"])} != {len(batch_dict["passages"])}'

        if self.args.do_kd_biencoder:
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            batch_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid][doc_id] for doc_id in input_doc_ids[idx:idx + step_size]]
                batch_dict['kd_labels'].append(cur_kd_labels)
            assert len(batch_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(batch_dict['kd_labels']), len(examples['query_id']))

        return batch_dict


class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self) -> BiencoderDataset:
        train_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = BiencoderDataset(args=self.args, tokenizer=self.tokenizer, input_files=train_input_files)

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset
