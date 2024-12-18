import math
from enum import Enum
from numbers import Number
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


from transformers import mpu


def get_entropy(gen_logits: torch.Tensor, inf_mask: torch.Tensor, mask: torch.Tensor, model_parallel: bool = False):
    inf_mask = torch.isinf(gen_logits) | inf_mask
    if model_parallel:
        full_probs = mpu.parallel_softmax(gen_logits.float(), dim=-1)
        full_logprobs = mpu.parallel_log_softmax(gen_logits.float(), dim=-1)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
        ent = -mpu.parallel_sum(full_probs * full_logprobs, dim=-1)
    else:
        full_probs = F.softmax(gen_logits, dim=-1, dtype=torch.float32)
        full_logprobs = F.log_softmax(gen_logits, dim=-1, dtype=torch.float32)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)        
        ent = -torch.sum(full_probs * full_logprobs, dim=-1)
    ent = ent * mask    
    return ent


def get_log_probs(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, inf_mask: torch.Tensor = None, model_parallel: bool = False):
    if model_parallel:
        logprobs = -mpu.parallel_logprobs(logits, ids)
        if inf_mask is not None:
            gathered_inf_mask = mpu.parallel_gather(inf_mask, -1, ids.unsqueeze(-1)).squeeze(-1)
            logprobs = logprobs.masked_fill(gathered_inf_mask, -float("inf"))
    else:
        logprobs = F.log_softmax(logits, dim=-1)
        if inf_mask is not None:
            logprobs = logprobs.masked_fill(inf_mask, -float("inf"))
        logprobs = torch.gather(logprobs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    logprobs = logprobs.masked_fill(~(mask.bool()), 0)
    
    # we ensure that the selected logprobs are not inf or nan
    assert all((~torch.isinf(logprobs.view(-1))) & (~torch.isnan(logprobs.view(-1))))
    
    return logprobs


def get_x_entropy(logits_1: torch.Tensor, logits_2: torch.Tensor, inf_mask: torch.Tensor, mask: torch.Tensor, model_parallel: bool = False):
    inf_mask = torch.isinf(logits_1) | torch.isinf(logits_2) | inf_mask
    if model_parallel:
        full_probs = mpu.parallel_softmax(logits_1.float(), dim=-1)
        full_logprobs = mpu.parallel_log_softmax(logits_2.float(), dim=-1)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
        xent = -mpu.parallel_sum(full_probs * full_logprobs, dim=-1)
    else:
        full_probs = F.softmax(logits_1, dim=-1, dtype=torch.float32)
        full_logprobs = F.log_softmax(logits_2, dim=-1, dtype=torch.float32)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
        xent = -torch.sum(full_probs * full_logprobs, dim=-1)
    xent = xent * mask
    return xent


def get_rev_kl(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor):
    log_ratio = (log_p - log_q) * mask
    kl = log_ratio.float().exp() - 1 - log_ratio
    return kl


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean: bool = True, distributed: bool = True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def significant(x: Number, ndigits: int = 2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))
