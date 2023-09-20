import torch

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from config import Arguments


@dataclass
class BiencoderCollator:

    args: Arguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        queries: List[str] = [f['query'] for f in features]
        passages: List[str] = sum([f['passages'] for f in features], [])

        input_texts = queries + passages

        merged_batch_dict = self.tokenizer(
            input_texts,
            max_length=self.args.max_len,
            truncation=True,
            padding=self.padding,
            return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(queries), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels
        return merged_batch_dict
