import torch
from utils import print_rank
    

class ToyLARSOptimizer(torch.optim.Optimizer):
    '''
        Toy implementation of LARS optimizer
        Assuming Params are list of tensors
    '''
    
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super(ToyLARSOptimizer, self).__init__(params, defaults)
        
    def step(self):
        params = self.param_groups[0]["params"]
        for l, p in enumerate(params):
            if p.grad is None:
                continue
            grad = p.grad.data
            weight_norm = torch.norm(p.data)
            grad_norm = torch.norm(grad)
            local_lr = self.param_groups[0]["lr"] * weight_norm / (grad_norm + 1e-8)
            p.data.add_(grad, alpha=-local_lr)