import torch
import torch.nn as nn

from abc import abstractmethod
from typing import List, Optional, Union


class BaseLLM(nn.Module):

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def batch_score(self, input_texts: List[str], output_texts: List[str], **kwargs) -> List[float]:
        raise NotImplementedError

    def score(self, input_text: str, output_text: str, **kwargs) -> float:
        return self.batch_score([input_text], [output_text], **kwargs)[0]

    @abstractmethod
    def batch_decode(self, input_texts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError

    def decode(self, input_text: str, **kwargs) -> str:
        return self.batch_decode([input_text], **kwargs)[0]

    def cuda(self, device: Optional[Union[int, torch.device]] = 0):
        self.model.to(device)
        return self
