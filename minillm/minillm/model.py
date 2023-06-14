import torch.nn as nn
from transformers import (
    AutoConfig,)

from .utils import get_model


class PPOModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_type: str,
        model_parallel=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.model_parallel = model_parallel
        self.config = AutoConfig.from_pretrained(model_path)
        self.base_model = get_model(model_path, model_type, model_parallel, gradient_checkpointing)

    def forward(self, **x):
        base_model_outputs = self.base_model(**x)
        return base_model_outputs
    
    def generate(self, **x):
        return self.base_model.generate(**x)
    
    def set_force_gradient_checkpointing(self, value):
        self.base_model.set_force_gradient_checkpointing(value)
