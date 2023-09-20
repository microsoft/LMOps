from typing import List, Dict
from datasets import Dataset

from config import Arguments


def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def filter_invalid_examples(args: Arguments, dataset: Dataset) -> Dataset:
    def _filter_func(example: Dict) -> bool:
        if len(example['doc_ids']) <= args.topk_as_positive:
            return False
        if example['task_name'] in args.held_out_tasks:
            return False

        sorted_doc_scores = sorted(example['doc_scores'], reverse=True)
        if sorted_doc_scores[args.topk_as_positive - 1] <= -100.:
            return False

        return True

    return dataset.filter(
        _filter_func,
        load_from_cache_file=args.world_size > 1,
    )


def group_doc_ids(examples: Dict[str, List],
                  negative_size: int,
                  offset: int) -> List[int]:
    pos_doc_ids: List[int] = []
    positives: List[Dict[str, List]] = examples['positives']
    for idx, ex_pos in enumerate(positives):
        all_pos_doc_ids = ex_pos['doc_id']
        cur_pos_doc_id = _slice_with_mod(all_pos_doc_ids, offset=offset, cnt=1)[0]
        pos_doc_ids.append(int(cur_pos_doc_id))

    neg_doc_ids: List[List[int]] = []
    negatives: List[Dict[str, List]] = examples['negatives']
    for ex_neg in negatives:
        cur_neg_doc_ids = _slice_with_mod(ex_neg['doc_id'],
                                          offset=offset * negative_size,
                                          cnt=negative_size)
        cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
        neg_doc_ids.append(cur_neg_doc_ids)

    assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
        input_doc_ids.append(pos_doc_id)
        input_doc_ids += neg_ids

    return input_doc_ids
