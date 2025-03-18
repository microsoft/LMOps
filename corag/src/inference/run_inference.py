import os
import torch
import copy
import json
import threading
import logging

from transformers import HfArgumentParser, AutoTokenizer, PreTrainedTokenizerFast
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from datasets import Dataset, load_dataset

from config import Arguments
from logger_config import logger
from data_utils import log_random_samples, load_corpus, format_documents_for_final_answer
from vllm_client import VllmClient, get_vllm_model_id
from utils import save_json_to_file, AtomicCounter
from agent import CoRagAgent, RagPath
from inference.metrics import compute_metrics_dict

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
logger.info('Args={}'.format(str(args)))

vllm_client: VllmClient = VllmClient(get_vllm_model_id())
corpus: Dataset = load_corpus()
corag_agent: CoRagAgent = CoRagAgent(vllm_client=vllm_client, corpus=corpus)
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(get_vllm_model_id())
tokenizer_lock: threading.Lock = threading.Lock()
processed_cnt: AtomicCounter = AtomicCounter()
total_cnt: int = 0


def _generate_single_example(ex: Dict) -> Dict:
    # Input columns: query / query_id / answers / context_doc_ids / context_doc_scores
    # Add following columns to the output: subqueries / subanswers / path_doc_ids
    if args.decode_strategy == 'greedy' or args.max_path_length < 1:
        path: RagPath = corag_agent.sample_path(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=args.max_path_length,
            temperature=0., max_tokens=64
        )
    elif args.decode_strategy == 'tree_search':
        path: RagPath = corag_agent.tree_search(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature, max_tokens=64
        )
    elif args.decode_strategy == 'best_of_n':
        path: RagPath = corag_agent.best_of_n(
            query=ex['query'], task_desc=ex['task_desc'],
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature,
            n = args.best_n,
            max_tokens=64
        )
    else:
        raise ValueError(f'Invalid decode strategy: {args.decode_strategy}')

    documents: List[str] = format_documents_for_final_answer(
        args=args,
        context_doc_ids=ex['context_doc_ids'],
        tokenizer=tokenizer, corpus=corpus,
        lock=tokenizer_lock
    )

    prediction: str = corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc=ex['task_desc'],
        documents=documents,
        max_message_length=args.max_len,
        temperature=0., max_tokens=128
    )

    ex_with_path = copy.deepcopy(ex)
    ex_with_path['subqueries'] = path.past_subqueries
    ex_with_path['subanswers'] = path.past_subanswers
    ex_with_path['path_doc_ids'] = path.past_doc_ids
    if 'title' in corpus.column_names:
        ex_with_path['path_doc_titles'] = [
            [corpus[int(doc_id)]['title'] for doc_id in doc_ids] for doc_ids in path.past_doc_ids
        ]
    ex_with_path['prediction'] = prediction

    processed_cnt.increment()
    if processed_cnt.value % 10 == 0:
        logger.info(
            f'Processed {processed_cnt.value} / {total_cnt} examples, '
            f'average token consumed: {vllm_client.token_consumed.value / processed_cnt.value:.2f}'
        )

    return ex_with_path


@torch.no_grad()
def main():
    if args.max_path_length < 1:
        logger.info('max_path_length < 1, setting decode_strategy to greedy')
        args.decode_strategy = 'greedy'

    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=args.num_threads)
    out_path: str = f'{args.output_dir}/preds_{args.decode_strategy}_{args.eval_task}_{args.eval_split}.jsonl'
    if os.path.exists(out_path):
        logger.info(f'{out_path} already exists. Skipping...')
        return

    logger.info(f'Processing {args.eval_task}-{args.eval_split}...')
    ds: Dataset = load_dataset('corag/multihopqa', args.eval_task, split=args.eval_split)
    ds = ds.remove_columns([name for name in ['subqueries', 'subanswers', 'predictions'] if name in ds.column_names])
    ds = ds.add_column('task_desc', ['answer multi-hop questions' for _ in range(len(ds))])

    if args.dry_run:
        ds = ds.select(range(16))
    log_random_samples(ds)
    global total_cnt
    total_cnt = len(ds)

    results: List[Dict] = list(executor.map(_generate_single_example, ds))
    ds = ds.add_column('prediction', [r['prediction'] for r in results])
    ds = ds.add_column('subqueries', [r['subqueries'] for r in results])
    ds = ds.add_column('subanswers', [r['subanswers'] for r in results])

    predictions: List[str] = ds['prediction']
    answers: List[List[str]] = ds['answers']
    metric_dict: Dict = compute_metrics_dict(
        labels=answers, preds=predictions, eval_metrics=args.eval_metrics
    )
    metric_dict['num_samples'] = len(ds)
    metric_dict['eval_task'] = args.eval_task
    metric_dict['eval_split'] = args.eval_split
    metric_dict['max_path_length'] = args.max_path_length
    metric_dict['decode_strategy'] = args.decode_strategy
    metric_dict['token_consumed'] = vllm_client.token_consumed.value
    metric_dict['average_token_consumed_per_sample'] = vllm_client.token_consumed.value / len(ds)
    logger.info('eval metric for input {}-{}: {}'.format(
        args.eval_task, args.eval_split, json.dumps(metric_dict, indent=4, ensure_ascii=False))
    )
    save_json_to_file(metric_dict, path=f'{args.output_dir}/metrics_{args.eval_task}_{args.eval_split}_{args.decode_strategy}.json')

    ds = ds.remove_columns([
        name for name in ['context_doc_ids', 'context_doc_scores'] if name in ds.column_names
    ])

    save_json_to_file(ds, path=out_path, line_by_line=True)
    logger.info(f'Saved predictions to {out_path}')
    logger.info('Done!')


if __name__ == '__main__':
    main()
