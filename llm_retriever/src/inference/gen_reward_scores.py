import collections
import os
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
)

from config import Arguments
from logger_config import logger
from utils import move_to_device, save_dataset, wait_until_all_files_show_up
from models import RerankerForInference
from data_utils import load_corpus, save_to_readable_format
from inference.inference_utils import reward_transform_func

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
kd_gen_score_in_path = os.path.join(args.data_dir, '{}.jsonl.gz'.format(args.kd_gen_score_split))
kd_gen_score_out_path = os.path.join(args.output_dir, 'kd_{}.jsonl.gz'.format(args.kd_gen_score_split))


def _get_shard_path(worker_idx: int) -> str:
    basename = os.path.basename(kd_gen_score_in_path)
    return '{}/shard_{}_{}'.format(args.output_dir, worker_idx, basename)


@torch.no_grad()
def _worker_gen_teacher_score():
    gpu_idx: int = args.process_index
    dataset = load_dataset('json', data_files=kd_gen_score_in_path)['train']
    if args.dry_run:
        dataset = dataset.select(range(100))
    dataset = dataset.shard(num_shards=args.world_size,
                            index=gpu_idx,
                            contiguous=True)

    qid_pids = []
    for ex in tqdm.tqdm(dataset, desc='get qid-pid pairs', mininterval=3):
        for doc_id in ex['doc_ids']:
            qid_pids.append((ex['query_id'], doc_id, ex['query'], ex['answers'], ex['options']))

    inference_dataset = Dataset.from_dict({
        'query_id': [t[0] for t in qid_pids],
        'doc_id': [t[1] for t in qid_pids],
        'query': [t[2] for t in qid_pids],
        'answers': [t[3] for t in qid_pids],
        'options': [t[4] for t in qid_pids],
    })

    query_ids, doc_ids = inference_dataset['query_id'], inference_dataset['doc_id']

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(inference_dataset)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: RerankerForInference = RerankerForInference.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(gpu_idx)

    corpus: Dataset = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))
    inference_dataset.set_transform(partial(reward_transform_func, tokenizer, args.reward_max_length, corpus))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    data_loader = DataLoader(
        inference_dataset,
        batch_size=args.kd_gen_score_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    scores = []
    for batch_dict in tqdm.tqdm(data_loader, desc='generate teacher score', mininterval=5):
        batch_dict = move_to_device(batch_dict, device=gpu_idx)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: SequenceClassifierOutput = model(batch_dict)
        scores.append(outputs.logits.squeeze(dim=-1).cpu())
        assert len(scores[-1].shape) == 1

    all_scores = torch.cat(scores, dim=-1)
    assert all_scores.shape[0] == len(inference_dataset), '{} != {}'.format(all_scores.shape[0], len(inference_dataset))
    all_scores = all_scores.tolist()

    query_id_to_doc_id_scores: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for idx in range(len(query_ids)):
        query_id_to_doc_id_scores[query_ids[idx]].append((doc_ids[idx], round(all_scores[idx], 5)))

    def _update_score(ex: Dict) -> Dict:
        query_id = ex['query_id']
        ex['doc_ids'] = [t[0] for t in query_id_to_doc_id_scores[query_id]]
        ex['doc_scores'] = [t[1] for t in query_id_to_doc_id_scores[query_id]]
        return ex

    dataset = dataset.map(_update_score, batched=False)
    save_dataset(dataset, _get_shard_path(gpu_idx))

    logger.info('Done computing teacher score for worker {}'.format(gpu_idx))


def _merge_teacher_scores():
    wait_until_all_files_show_up([_get_shard_path(worker_idx) for worker_idx in range(args.world_size)])

    dataset = load_dataset(
        'json', data_files=[_get_shard_path(worker_idx) for worker_idx in range(args.world_size)], split='train'
    )

    save_dataset(dataset, kd_gen_score_out_path)
    logger.info('Writing teacher score to {}'.format(kd_gen_score_out_path))

    logger.info('Done merge results')

    corpus = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))
    save_to_readable_format(in_path=kd_gen_score_out_path, corpus=corpus, shuffle=True)

    for worker_idx in range(args.world_size):
        os.remove(_get_shard_path(worker_idx))


def main():
    logger.info('Args={}'.format(str(args)))
    if os.path.exists(kd_gen_score_out_path):
        logger.info('Found {}, skip'.format(kd_gen_score_out_path))
        return

    logger.info('Use {} workers'.format(args.world_size))
    _worker_gen_teacher_score()
    logger.info('Done batch generate teacher score')

    if args.process_index <= 0:
        _merge_teacher_scores()


if __name__ == '__main__':
    main()
