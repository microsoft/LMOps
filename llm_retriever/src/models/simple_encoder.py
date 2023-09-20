import torch
import torch.nn.functional as F
import tqdm

from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict

from utils import pool, move_to_cuda


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List],
                    prompt: str = None) -> BatchEncoding:
    if prompt:
        examples['input_texts'] = [prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=256,
        return_token_type_ids=False,
        padding=True,
        truncation=True,
    )

    return batch_dict


class SimpleEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str,
                 l2_normalize: bool = True,
                 pool_type: str = 'avg',
                 prompt: str = 'query: '):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.l2_normalize = l2_normalize
        self.pool_type = pool_type
        self.prompt = prompt
        assert self.prompt in ['', 'query: ', 'passage: ']

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> torch.Tensor:
        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer, prompt=self.prompt))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=128 * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu())

        return torch.cat(encoded_embeds, dim=0)
