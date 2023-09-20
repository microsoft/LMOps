import os
import glob
import json
import time
import tqdm
import gzip
import torch
import torch.distributed as dist

from torch import Tensor
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from typing import List, Union, Optional, Tuple, Mapping, Dict

from logger_config import logger


def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')


def save_dataset(dataset: Dataset, out_path: str, shuffle: bool = False):
    assert out_path.endswith('.jsonl.gz') or out_path.endswith('.jsonl'), 'Only jsonl format is supported'
    if shuffle:
        dataset = dataset.shuffle(seed=123)

    if out_path.endswith('.jsonl.gz'):
        f = gzip.open(out_path, 'wt', encoding='utf-8', compresslevel=1)
    else:
        f = open(out_path, 'w', encoding='utf-8')

    column_names = dataset.column_names
    examples_per_shard = 10_000_000
    for start_idx in tqdm.tqdm(range(0, len(dataset), examples_per_shard),
                               desc='Saving dataset to {}'.format(out_path)):
        end_idx = min(start_idx + examples_per_shard, len(dataset))
        sub_dataset = dataset.select(range(start_idx, end_idx))
        column_name_to_data: dict = {column_name: sub_dataset[column_name] for column_name in column_names}
        for idx in range(len(sub_dataset)):
            example = {k: column_name_to_data[k][idx] for k in column_names}
            f.write(json.dumps(example, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
    f.close()


def move_to_device(sample, device: Union[int, torch.device]):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_device(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_device(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_device(sample)


def move_to_cuda(sample):
    return move_to_device(sample, device=0)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def dist_gather_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None

    x = x.contiguous()
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


@torch.no_grad()
def select_grouped_indices(scores: torch.Tensor,
                           group_size: int,
                           start: int = 0) -> torch.Tensor:
    assert len(scores.shape) == 2
    batch_size = scores.shape[0]
    assert batch_size * group_size <= scores.shape[1]

    indices = torch.arange(0, group_size, dtype=torch.long)
    indices = indices.repeat(batch_size, 1)
    indices += torch.arange(0, batch_size, dtype=torch.long).unsqueeze(-1) * group_size
    indices += start

    return indices.to(scores.device)


def full_contrastive_scores_and_labels(query: torch.Tensor,
                                       key: torch.Tensor,
                                       use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    kq.fill_diagonal_(float('-inf'))

    if not use_all_pairs:
        return torch.cat([qk, kq], dim=-1), labels

    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name: str, round_digits: int = 3):
        self.name = name
        self.round_digits = round_digits
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{}: {}'.format(self.name, round(self.avg, self.round_digits))


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def get_input_files(input_files_pattern: str) -> List[str]:
    input_files: List[str] = []

    for pattern in input_files_pattern.strip().split(','):
        input_files += sorted(glob.glob(pattern))

    input_files = sorted(list(set(input_files)))
    logger.info('{} input files: {}'.format(len(input_files), input_files))
    return input_files


class DictTrie:
    def __init__(self, sequences: List[List[int]], bos_token_id: int):
        self.trie_dict = {}
        self.len = 0
        for sequence in sequences:
            if bos_token_id == sequence[0]:
                sequence = sequence[1:]
            DictTrie._add_to_trie(sequence, self.trie_dict)
            self.len += 1

    def get(self, prefix_sequence: List[int]):
        return DictTrie._get_from_trie(prefix_sequence, self.trie_dict)

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            DictTrie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return DictTrie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]]
            )
        else:
            return []

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def build_trie(tokenizer: PreTrainedTokenizerFast, output_texts: List[str]) -> DictTrie:
    # llama tokenizer will produce 3 IDs...
    newline_id = tokenizer.encode('\n')[-1]
    output_texts = ['\n{}'.format(t) for t in output_texts]

    # we assume the outputs won't be too long
    output_ids: List[List[int]] = tokenizer(
        output_texts, max_length=512, truncation=True, padding=False
    )['input_ids']

    for idx in range(len(output_ids)):
        newline_idx = output_ids[idx].index(newline_id)
        assert newline_idx >= 0
        output_ids[idx] = output_ids[idx][(newline_idx + 1):]
        assert len(output_ids[idx]) > 0, output_texts[idx]

    return DictTrie(output_ids, bos_token_id=tokenizer.bos_token_id)


def wait_until_all_files_show_up(file_names: List[str]):
    # TODO: this is a hacky way, since a file may be created but not fully written
    while True:
        time.sleep(5)
        all_files_exist = True
        for file_name in file_names:
            if not os.path.exists(file_name):
                all_files_exist = False
                break
        if all_files_exist:
            time.sleep(30)
            return
