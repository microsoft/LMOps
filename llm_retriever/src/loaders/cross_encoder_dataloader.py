import os.path
import random
import torch

from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from utils import get_input_files
from .loader_utils import group_doc_ids, filter_invalid_examples
from data_utils import to_positive_negative_format


class CrossEncoderDataset(torch.utils.data.Dataset):

    def __init__(self, input_files: List[str], args: Arguments,
                 tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.input_files = input_files
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        corpus_path = os.path.join(os.path.dirname(self.input_files[0]), 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path, split='train')

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
        current_epoch = int(self.trainer.state.epoch or 0) if self.trainer is not None else 0

        examples = deepcopy(examples)
        # add some random negatives if not enough
        for idx in range(len(examples['query_id'])):
            while len(examples['negatives'][idx]['doc_id']) < self.negative_size:
                random_doc_id = str(random.randint(0, len(self.corpus) - 1))
                examples['negatives'][idx]['doc_id'].append(random_doc_id)
                examples['negatives'][idx]['score'].append(-100.)

        input_doc_ids = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        input_queries, input_docs = [], []
        for idx, doc_id in enumerate(input_doc_ids):
            input_docs.append(self.corpus[doc_id]['contents'].strip())
            # For reward model, the left side is the query + ground truth answer
            q_idx = idx // self.args.train_n_passages
            current_query: str = examples['query'][q_idx]
            answers, options = examples['answers'][q_idx], examples['options'][q_idx]
            if len(options) > 1:
                current_query += '\n' + options[ord(answers[0]) - ord('A')]
                # logger.info('current_query: %s', current_query)
            else:
                current_query += '\n' + random.choice(answers)
            input_queries.append(current_query)

        batch_dict = self.tokenizer(input_queries,
                                    text_pair=input_docs,
                                    max_length=self.args.reward_max_length,
                                    return_token_type_ids=False,
                                    padding=PaddingStrategy.DO_NOT_PAD,
                                    truncation=True)

        packed_batch_dict = {}
        for k in batch_dict:
            packed_batch_dict[k] = []
            assert len(examples['query']) * self.args.train_n_passages == len(batch_dict[k])
            for idx in range(len(examples['query'])):
                start = idx * self.args.train_n_passages
                packed_batch_dict[k].append(batch_dict[k][start:(start + self.args.train_n_passages)])

        return packed_batch_dict


class CrossEncoderDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self) -> CrossEncoderDataset:
        train_dataset = None

        if self.args.train_file is not None:
            train_input_files = get_input_files(self.args.train_file)
            logger.info("Train files: {}".format(train_input_files))
            train_dataset = CrossEncoderDataset(
                args=self.args,
                tokenizer=self.tokenizer,
                input_files=train_input_files,
            )

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset
