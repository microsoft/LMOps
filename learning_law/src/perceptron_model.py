import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


class PerceptronModel(nn.Module):
    def __init__(self, config):
        super(PerceptronModel, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.linear = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
    
    def compute_loss(self, x, y, gamma=None):
        logits = self.forward(x)
        logits = logits.squeeze()
        y = y.squeeze()
        losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
        if gamma is not None:
            loss = torch.sum(losses * gamma)
        else:
            loss = torch.mean(losses)
        
        return loss, losses
    
    @staticmethod
    def compute_loss_func(params, buffers, model, x, y, gamma=None):
        logits = functional_call(model, (params, buffers), x)
        logits = logits.squeeze()
        y = y.squeeze()
        losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
        if gamma is not None:
            loss = torch.sum(losses * gamma)
        else:
            loss = torch.mean(losses)
        return loss

    @staticmethod
    def compute_loss_func_single(params, buffers, model, x, y):
        y = y.squeeze()
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        logits = functional_call(model, (params, buffers), x)
        logits = logits.squeeze()
        logits = logits.unsqueeze(0)
        losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
        loss = torch.mean(losses)
        return loss

    def vector_to_params(self, vec):
        pointer = 0
        d = {}
        for n, p in self.named_parameters():
            d[n] = nn.Parameter(vec[pointer:pointer+p.numel()].view(p.size()))
            pointer += p.numel()
        
        return d
    
    def params_to_vector(self, params):
        vec = []
        for n, p in self.named_parameters():
            vec.append(params[n].view(-1))
        vec = torch.cat(vec, dim=0)
        return vec