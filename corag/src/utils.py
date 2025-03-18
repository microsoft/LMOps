import os
import json
import torch
import threading

from torch import Tensor
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding
from typing import List, Union, Mapping, Dict, Any

from logger_config import logger


def save_json_to_file(objects: Union[List, dict, Dataset], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list) or isinstance(objects, Dataset), 'objects should be a list or Dataset'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')


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


def batch_truncate(
        texts: List[str], tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128, truncate_from_middle: bool = False,
        skip_special_tokens: bool = False
) -> List[str]:
    if not texts:
        return texts

    if truncate_from_middle:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=int(10**7))['input_ids']
        for idx in range(len(input_ids)):
            if len(input_ids[idx]) > max_length:
                half_length = max_length // 2
                input_ids[idx] = input_ids[idx][:half_length] + input_ids[idx][-half_length:]
    else:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=max_length)['input_ids']

    # Skip bos but keep other special tokens in the input texts
    if not skip_special_tokens:
        input_ids = [ids[1:] if ids[0] == tokenizer.bos_token_id else ids for ids in input_ids]

    return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)


def log_truncate(value: Any, max_chars: int = 2048, max_items: int = 4) -> str:
    def _maybe_truncate(v):
        if isinstance(v, str):
            return v[:max_chars // 2] + "\n...\n" + v[-max_chars // 2:] if len(v) > max_chars else v
        elif isinstance(v, list):
            v = v[:max_items // 2] + ['...'] + v[-max_items // 2:] if len(v) > max_items else v
            return [_maybe_truncate(x) for x in v]
        elif isinstance(v, dict):
            return {k: _maybe_truncate(v) for k, v in v.items()}
        else:
            return v

    value = _maybe_truncate(value)
    if not isinstance(value, str):
        line: str = json.dumps(value, indent=2, ensure_ascii=False, default=lambda o: '<not serializable>')
    else:
        line = value

    return line[:max_chars // 2] + "\n...\n" + line[-max_chars // 2:] if len(line) > max_chars else line


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], max_length: int = 512) -> BatchEncoding:
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        return_tensors='pt'
    )


def get_task_def_by_task_name(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        # KILT eval instructions
        'aidayago2': 'Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity',
        'cweb': 'Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity',
        'eli5': 'Provided a user question, retrieve Wikipedia passages that can help answer the question',
        'fever': 'Given a claim, retrieve documents that support or refute the claim',
        'hotpotqa': 'Given a multi-hop question, retrieve documents that can help answer the question',
        'structured_zeroshot': 'Given a head entity and a relation, retrieve the Wikipedia page that contains the tail entity',
        'trex': 'Given a head entity and a relation, retrieve the Wikipedia page that contains the tail entity',
        'triviaqa': 'Given a question, retrieve Wikipedia passages that answer the question',
        'wned': 'Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity',
        'wow': 'Given a conversation history, retrieve a Wikipedia passage that contains relevant information to continue the conversation',
        'blink': 'Given a mention of named entity with surrounding context, retrieve the Wikipedia page that links to the entity',
    }

    if task_name in task_name_to_instruct:
        return task_name_to_instruct[task_name]

    logger.warning(f'Task name {task_name} not found in task_name_to_instruct, will use default instruct')
    return 'Given a search query, retrieve relevant documents that can help answer the query'


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


class AtomicCounter:
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

    def reset(self):
        self.value = 0
        return self.value
