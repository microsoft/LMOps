import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import json

from utils import get_model, print_rank, all_gather
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM


class BertBaseModel(nn.Module):
    def __init__(self, args, device, base_model_path):
        super().__init__()
        self.model_path = base_model_path
        dtype = torch.float32 if args.fp32 else torch.float16
        self.model = AutoModel.from_pretrained(self.model_path, device_map={"": device}, torch_dtype=dtype)
    
    def forward(self, input_ids, attention_mask, use_cache=False):
        output1 = self.model(input_ids=input_ids[:, :512], attention_mask=attention_mask[:, :512], return_dict=True)
        output2 = self.model(input_ids=input_ids[:, 512:], attention_mask=attention_mask[:, 512:], return_dict=True)
        h1 = output1["last_hidden_state"]
        h2 = output2["last_hidden_state"]
        
        h = torch.cat([h1, h2], dim=1)
        return {"last_hidden_state": h}


class DataScorerModel(nn.Module):
    def __init__(self, args, device, base_model_path, bias=False, encoding="mean", head_type="linear"):
        super().__init__()
        self.args = args
        self.base_model_path = base_model_path
        self.config = AutoConfig.from_pretrained(base_model_path)
        if self.args.model_type in ["bert", "roberta"]:
            self.base_model = BertBaseModel(args, device, base_model_path)
        else:
            self.base_model = get_model(args, device, base_model_path, self.config, model_cls=AutoModel)
        self.head = nn.Linear(self.config.hidden_size, 1, bias=bias)
        self.bias = bias
        self.head_type = head_type
        self.encoding = encoding
        
        print_rank(f"Data Scorer | Bias: {bias}, Encoding: {encoding}, Head type: {head_type}")

    def _forward(self, input_ids, attention_mask, pos):
        h = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)["last_hidden_state"]
        if self.encoding == "mean":
            mask = (torch.arange(h.size(1), device=h.device)[None, :] <= pos[:, None]).to(h.dtype)
            origin_dtype = h.dtype
            h = h.float()
            h = torch.sum(h * mask[:, :, None], dim=1) / mask.sum(dim=1)[:, None]
            h = h.to(origin_dtype)            
        elif self.encoding == "last":
            h = torch.gather(h, 1, pos[:, None, None].expand(-1, -1, h.size(-1))).squeeze()
        elif self.encoding == "first":
            h = h[:, 0]
        else:
            raise ValueError("encoding should be mean or last")
        
        if self.head_type == "linear":
            s = self.head(h).squeeze()
        elif self.head_type == "sigmoid":
            s = torch.sigmoid(self.head(h).squeeze())
        else:
            raise ValueError("score_head should be linear or sigmoid")

        if s.dim() == 0:
            s = s.unsqueeze(0)

        return s.float()

    def forward(self, input_ids, attention_mask, pos, labels):
        s = self._forward(input_ids, attention_mask, pos)
        labels = labels.to(s.dtype)  # (b, s), B
        loss = torch.nn.functional.mse_loss(s, labels)
        return loss

    def save_pretrained(self, save_dir, **kawrgs):
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({
                "base_model_path": self.base_model_path.replace(self.args.base_path, ""),
                "bias": self.bias,
                "encoding": self.encoding
            }, f, indent=4)
        torch.save(self.state_dict(), os.path.join(
            save_dir, "data_scorer_model.pt"))
    
    def inference(self, input_ids, attention_mask, pos, labels=None):
        return self._forward(input_ids, attention_mask, pos)
