import torch.nn as nn
from transformers import (
    AutoConfig,)

from utils import get_model


class PPOModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model_parallel = args.model_parallel
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.base_model = get_model(args, device)
        self.base_model.eval() # no dropout for RL

    def forward(self, **x):
        base_model_outputs = self.base_model(**x)
        return base_model_outputs
    
    def generate(self, **x):
        return self.base_model.generate(**x)
    
    def set_force_gradient_checkpointing(self, value):
        self.base_model.set_force_gradient_checkpointing(value)
