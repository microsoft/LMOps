import torch
import torch.nn as nn
from torch.func import functional_call
from utils import print_rank


class TransformerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids, attention_mask)
    
    def compute_loss(self, input_ids, attention_mask, label, loss_mask):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = self.forward(input_ids, attention_mask).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(label.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(losses)

        return loss, losses
    
    @staticmethod
    def compute_loss_func(params, buffers, model, input_ids, attention_mask, label, loss_mask):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), (input_ids, attention_mask)).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(label.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(losses)
        return loss
    
    @staticmethod
    def compute_loss_func_single(params, buffers, model, input_ids, attention_mask, label, loss_mask):
        input_ids = input_ids.unsqueeze(0)
        label = label.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        loss_mask = loss_mask.unsqueeze(0)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(model, (params, buffers), (input_ids, attention_mask)).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(label.size(0), -1)
        losses = torch.sum(losses * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(losses)
        return loss

    def vector_to_params(self, vec):
        pointer = 0
        d = {}
        for n, p in self.named_parameters():
            d[n] = nn.Parameter(vec[pointer:pointer+p.numel()].view(p.size()), requires_grad=False)
            pointer += p.numel()
        
        assert pointer == vec.numel()

        return d

    def params_to_vector(self, params):
        vec = []
        for n, p in self.named_parameters():
            vec.append(params[n].view(-1))
        vec = torch.cat(vec, dim=0)
        return vec

    def get_params_vec(self):
        vec = []
        for n, p in self.named_parameters():
            vec.append(p.view(-1))
        vec = torch.cat(vec, dim=0)
        return vec
    
    def set_params_vec(self, vec):
        params = self.vector_to_params(vec)
        for n, p in self.named_parameters():
            p.data = params[n].data