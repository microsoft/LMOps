import torch

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from logger_config import logger


@dataclass
class ScoreCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    delimiter: str = '\n'

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        self.tokenizer.padding_side = 'right'
        input_texts = [f['input_texts'] for f in features]
        output_texts = [f['output_texts'] for f in features]
        assert all(not text.endswith(self.delimiter) for text in input_texts)
        assert all(not text.startswith(self.delimiter) for text in output_texts)
        concat_texts: List[str] = [self.delimiter.join([inp, out]) for inp, out in zip(input_texts, output_texts)]

        batch_dict = self.tokenizer(
            concat_texts,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        labels = batch_dict['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        num_valid_tokens = torch.cumsum(batch_dict['attention_mask'], dim=1)
        output_lengths: torch.LongTensor = torch.LongTensor(self._get_output_lengths(output_texts))
        logger.debug('output lengths: {}'.format(output_lengths))
        input_lengths: torch.LongTensor = torch.sum(batch_dict['attention_mask'], dim=1) - output_lengths
        labels[num_valid_tokens <= input_lengths[:, None]] = -100
        batch_dict['labels'] = labels

        return batch_dict

    def _get_output_lengths(self, output_texts: List[str]) -> List[int]:
        output_ids: List[List[int]] = self.tokenizer(
            output_texts, max_length=self.max_length, truncation=True, padding=False
        )['input_ids']

        for idx in range(len(output_ids)):
            # llama tokenizer prepend a bos token
            if output_ids[idx][0] == self.tokenizer.bos_token_id:
                output_ids[idx] = output_ids[idx][1:]

        lengths: List[int] = [len(output_id) for output_id in output_ids]
        assert all(length > 0 for length in lengths), lengths

        return lengths


@dataclass
class DecodeCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        # batch_score requires right padding, but generate requires left padding
        self.tokenizer.padding_side = 'left'
        input_texts = [f['input_texts'] for f in features]

        batch_dict = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        return batch_dict
