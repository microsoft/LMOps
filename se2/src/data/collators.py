import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedModel
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BatchEncoding, PreTrainedTokenizerBase
# from src.dataset_readers.qdmr_indexer import QDMRIndexerDatasetReader
from transformers.data.data_collator import DataCollatorWithPadding
from src.utils.dataset_utils import pad2sameLen


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data
    
    def to(self, device):
        return self.data


@dataclass
class DataCollatorWithPaddingAndCuda:

    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
    

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        metadata = [x.pop("metadata") for x in features]
        features = {key: [example[key] for example in features] for key in features[0].keys()}
        left_pad=(self.tokenizer.padding_side=="left")
        features={key: pad2sameLen(features[key],pad_idx=self.tokenizer.pad_token_id if 'input_ids' in key else 0, left_pad=left_pad) for key in features.keys()}
        # for key in features.keys(): dic[key]=pad2sameLen(features[key])
        batch=BatchEncoding(features,tensor_type="pt")
        batch['metadata'] = ListWrapper(metadata)
        if self.device:
            batch = batch.to(self.device)
        return batch
