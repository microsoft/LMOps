import os
import sys
import numpy as np

sys.path.insert(0, 'src/')

from datasets import Dataset, load_dataset, DownloadMode
from typing import Dict, List
from collections import Counter
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from utils import save_dataset, wait_until_all_files_show_up
from data_utils import save_to_readable_format
from llms import BaseLLM
from tasks import get_possible_answers_by_task_name
from model_utils import build_llm, parse_model_id

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def _get_shard_out_path(gpu_idx: int) -> str:
    return '{}/{}_{}_{}.jsonl.gz'.format(
        args.data_dir, os.path.basename(args.llm_model_name_or_path), args.search_split, gpu_idx)


def _select_indices(dataset: Dataset) -> List[int]:
    task_names: List[str] = dataset['task_name']
    num_examples_per_task: Dict[str, int] = Counter(task_names)

    left, right = 0, len(dataset)
    while left < right:
        middle = (left + right) // 2
        total_examples = sum([min(middle, v) for v in num_examples_per_task.values()])
        if total_examples >= args.max_train_samples:
            right = middle
        else:
            left = middle + 1

    indices: List[int] = []
    selected_per_task: Dict[str, int] = Counter()
    for idx, task_name in enumerate(task_names):
        if selected_per_task[task_name] < left:
            indices.append(idx)
            selected_per_task[task_name] += 1

    for task_name, num_selected in selected_per_task.items():
        logger.info('Task name: {}, selected {}/{}'.format(
            task_name, num_selected, num_examples_per_task[task_name])
        )

    return indices


def _worker_gen_llm_scores():
    data_path: str = '{}/{}.jsonl.gz'.format(args.data_dir, args.search_split)
    assert os.path.exists(data_path), 'Data file {} does not exist.'.format(data_path)
    dataset: Dataset = load_dataset(
        'json', data_files=data_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))
    elif args.max_train_samples is not None and 0 < args.max_train_samples < len(dataset):
        model_dependent_seed: int = args.seed + hash(args.llm_model_name_or_path) % 10 ** 6
        dataset = dataset.shuffle(seed=model_dependent_seed)
        indices: List[int] = _select_indices(dataset)
        dataset = dataset.select(indices).shuffle(seed=args.seed)

    gpu_idx: int = args.process_index
    dataset = dataset.shard(num_shards=args.world_size, index=gpu_idx, contiguous=True)
    logger.info('Worker {} needs to process {} examples'.format(gpu_idx, len(dataset)))

    corpus_path: str = '{}/passages.jsonl.gz'.format(args.data_dir)
    corpus: Dataset = load_dataset(
        'json', data_files=corpus_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    logger.info('Load {} candidates from {}'.format(len(corpus), corpus_path))

    llm: BaseLLM = build_llm(args)
    llm.cuda(gpu_idx)

    def _map_func(examples: Dict[str, List]) -> Dict[str, List]:
        # Only select a subset to score
        for i in range(len(examples['doc_ids'])):
            np.random.shuffle(examples['doc_ids'][i])
            examples['doc_ids'][i] = examples['doc_ids'][i][:args.search_topk]

        input_texts: List[str] = []
        output_texts: List[str] = []
        for query, doc_ids, answers, options in zip(examples['query'], examples['doc_ids'], examples['answers'], examples['options']):
            # Currently we use only the first answer.
            answer = answers[0]
            assert all(int(doc_id) >= 0 for doc_id in doc_ids)
            for doc_id in doc_ids:
                input_texts.append('{}\n\n{}'.format(corpus[int(doc_id)]['contents'], query))
                if len(options) <= 1:
                    output_texts.append(answer)
                else:
                    # multiple-choice tasks
                    assert answer in ['A', 'B', 'C', 'D'], 'Invalid answer: {}'.format(answer)
                    output_texts.append(options[ord(answer) - ord('A')])

        llm_scores: List[float] = llm.batch_score(input_texts=input_texts, output_texts=output_texts)

        start_idx = 0
        for i, doc_ids in enumerate(examples['doc_ids']):
            end_idx = start_idx + len(doc_ids)
            examples['doc_scores'][i] = llm_scores[start_idx:end_idx]
            start_idx = end_idx

        # if a retrieved doc comes from a different task, set its score to a very low value
        # for text classification task, if the answers are different, decrease its score
        for i, doc_ids in enumerate(examples['doc_ids']):
            q_task_name = examples['task_name'][i]
            for j, doc_id in enumerate(doc_ids):
                current_ex: dict = corpus[int(doc_id)]
                if current_ex['task_name'] != q_task_name:
                    examples['doc_scores'][i][j] += -100.

                if len(examples['options'][i]) > 1:
                    continue
                if not get_possible_answers_by_task_name(q_task_name) or current_ex['task_name'] != q_task_name:
                    continue

                answer = examples['answers'][i][0].strip()
                contents: str = current_ex['contents']
                input_ans: str = contents.strip().split('\n')[-1]
                if input_ans != answer:
                    examples['doc_scores'][i][j] += -100.

        return examples

    dataset = dataset.map(
        _map_func, batched=True, num_proc=1,
        desc='Worker {} compute LLM scores'.format(gpu_idx)
    )

    out_path = _get_shard_out_path(gpu_idx)
    save_dataset(dataset, out_path)
    logger.info('Worker {} saved {} examples to {}'.format(gpu_idx, len(dataset), out_path))


def _merge_scores(out_path: str):
    dataset: Dataset = load_dataset(
        'json', data_files=[_get_shard_out_path(gpu_idx) for gpu_idx in range(args.world_size)],
        split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )

    save_dataset(dataset, out_path)

    corpus_path: str = '{}/passages.jsonl.gz'.format(args.data_dir)
    corpus: Dataset = load_dataset(
        'json', data_files=corpus_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    save_to_readable_format(in_path=out_path, corpus=corpus, shuffle=True)


def main():
    logger.info('Args={}'.format(str(args)))
    out_path: str = '{}/{}_{}.jsonl.gz'.format(
        args.data_dir, parse_model_id(args.llm_model_name_or_path), args.search_split
    )
    if os.path.exists(out_path):
        logger.info('Output file {} exists. Skip.'.format(out_path))
        return

    logger.info('Use {} workers'.format(args.world_size))
    _worker_gen_llm_scores()

    wait_until_all_files_show_up([_get_shard_out_path(gpu_idx) for gpu_idx in range(args.world_size)])

    logger.info('All workers finished.')

    if args.process_index == 0:
        _merge_scores(out_path=out_path)
        for gpu_idx in range(args.world_size):
            os.remove(_get_shard_out_path(gpu_idx))
        os.system('rm -f {}/*.lock'.format(args.data_dir))


if __name__ == '__main__':
    main()
