import os
import sys
import numpy as np
sys.path.insert(0, 'src/')

from datasets import Dataset, load_dataset, DownloadMode
from typing import Dict, List, Tuple
from transformers import HfArgumentParser

from config import Arguments
from logger_config import logger
from utils import save_dataset
from data_utils import save_to_readable_format
from evaluation import BaseEval
from model_utils import build_eval_model, parse_model_id

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
assert args.do_search, 'This script is only for search mode.'


def main():
    model_id = parse_model_id(args.model_name_or_path)
    out_path: str = '{}/{}_{}.jsonl.gz'.format(args.output_dir, model_id, args.search_split)
    if os.path.exists(out_path):
        logger.info('Output file {} already exists. Skip.'.format(out_path))
        return

    data_path: str = '{}/{}.jsonl.gz'.format(args.data_dir, args.search_split)
    assert os.path.exists(data_path), 'Data file {} does not exist.'.format(data_path)
    dataset: Dataset = load_dataset(
        'json', data_files=data_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))
    logger.info('Load {} examples from {}'.format(len(dataset), data_path))

    corpus_path: str = '{}/passages.jsonl.gz'.format(args.data_dir)
    corpus: Dataset = load_dataset(
        'json', data_files=corpus_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    if args.dry_run:
        corpus = corpus.select(range(4096))
    logger.info('Load {} candidates from {}'.format(len(corpus), corpus_path))

    retriever: BaseEval = build_eval_model(args=args, corpus=corpus)

    logger.info('Search top {} candidates for each example.'.format(args.search_topk))
    topk_score_doc_ids: List[List[Tuple[float, str]]] = retriever.get_topk_score_doc_ids(
        dataset['query'], k=args.search_topk, task_names=dataset['task_name']
    )
    all_contents: List[str] = corpus['contents']

    def _map_func(example: Dict, idx: int) -> Dict:
        score_doc_ids: List[Tuple[float, str]] = topk_score_doc_ids[idx]
        # Exclude the example itself from the top-k candidates.
        score_doc_ids = [t for t in score_doc_ids if not all_contents[int(t[1])].startswith(example['query'])]
        np.random.shuffle(score_doc_ids)
        return {
            'doc_ids': [doc_id for _, doc_id in score_doc_ids],
            'doc_scores': [round(doc_score, 4) for doc_score, _ in score_doc_ids],
        }

    dataset = dataset.map(_map_func, with_indices=True, num_proc=1, desc='Add top-k candidates')
    dataset = dataset.filter(lambda example: len(example['doc_ids']) > 1)

    save_dataset(dataset, out_path, shuffle='train' in args.search_split)
    save_to_readable_format(in_path=out_path, corpus=corpus, shuffle=True)


if __name__ == '__main__':
    main()
