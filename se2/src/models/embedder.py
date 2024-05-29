from transformers import AutoModel
from typing import Dict
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name,cache_dir) -> None:
        super().__init__()
        self.embedder =  AutoModel.from_pretrained(model_name,cache_dir)

    def forward(self, input_ids, attention_mask,**kwargs) -> Dict[str, torch.Tensor]:
        enc_emb = self.embedder(input_ids)
        return mean_pooling(enc_emb, attention_mask)