import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import List, Dict, Optional, Union
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from logger_config import logger
from utils import pool, move_to_device, get_detailed_instruct, get_task_def_by_task_name, create_batch_dict
from search.model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE


class SimpleEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, prefix_type: Optional[str] = 'instruction', pool_type: Optional[str] = 'last',
                 task_name: Optional[str] = None, max_length: int = 512):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        base_name: str = model_name_or_path.split('/')[-1]
        self.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, pool_type)
        self.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, prefix_type)
        self.max_length = max_length
        assert self.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
        assert self.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'

        self.encoder = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.prompt: Optional[str] = None
        if self.prefix_type == 'instruction' and task_name is not None:
            task_def: str = get_task_def_by_task_name(task_name=task_name)
            self.prompt = get_detailed_instruct(task_def)
            logger.info('Set prompt: {}'.format(self.prompt))

        self.encoder.eval()
        logger.info(f'pool_type={self.pool_type}, prefix_type={self.prefix_type}, prompt={self.prompt}')

    def encode_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        return self._do_encode(input_texts, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dataset], **kwargs) -> torch.Tensor:
        input_texts = ['{}\n{}'.format(doc.get('title', ''), doc['contents']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if self.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts, **kwargs)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], **kwargs) -> torch.Tensor:
        encoded_embeds = []
        batch_size = kwargs.get('batch_size', 128)
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10, disable=len(input_texts) < 128):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, max_length=self.max_length)
            batch_dict = move_to_device(batch_dict, device=self.encoder.device)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().to(torch.float16))

        return torch.cat(encoded_embeds, dim=0)

    def to(self, device):
        self.encoder.to(device)
        return self
