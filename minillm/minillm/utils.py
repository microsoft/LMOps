import math
from enum import Enum
from numbers import Number
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from accelerate import init_empty_weights


from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    mpu,
    ParallelOPTForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,
    ParallelLlamaForCausalLM)


def get_entropy(gen_logits, inf_mask, mask, model_parallel=False):
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


def get_log_probs(logits, ids, mask, inf_mask=None, model_parallel=False):
    if model_parallel:
        logprobs = -mpu.parallel_logprobs(logits, ids)
        if inf_mask is not None:
            gathered_inf_mask = mpu.parallel_gather(inf_mask, -1, ids.unsqueeze(-1)).squeeze(-1)
            logprobs = logprobs.masked_fill(gathered_inf_mask, -float("inf"))
        # logprobs = mpu.parallel_log_softmax(logits.float(), dim=-1)
        # if inf_mask is not None:
        #     logprobs = logprobs.masked_fill(inf_mask, -float("inf"))
        # logprobs = mpu.parallel_gather(logprobs, -1, ids.unsqueeze(-1)).squeeze(-1)
    else:
        logprobs = F.log_softmax(logits, dim=-1)
        if inf_mask is not None:
            logprobs = logprobs.masked_fill(inf_mask, -float("inf"))
        logprobs = torch.gather(logprobs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    logprobs = logprobs.masked_fill(~(mask.bool()), 0)
    
    # we ensure that the selected logprobs are not inf or nan
    assert all((~torch.isinf(logprobs.view(-1))) & (~torch.isnan(logprobs.view(-1))))
    
    return logprobs


def get_x_entropy(logits_1, logits_2, inf_mask, mask, model_parallel=False):
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


def get_rev_kl(log_p, log_q, mask):
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


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


class OptimizerName(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    ADAM_8BIT_BNB: str = "adam_8bit_bnb"
    ADAMW_8BIT_BNB: str = "adamw_8bit_bnb"
    SGD: str = "sgd"


def get_optimizer_class(name: OptimizerName):
    """
    Returns the optimizer class with the given name

    Args:
        name (str): Name of the optimizer as found in `OptimizerNames`
    """
    if name == OptimizerName.ADAM:
        return torch.optim.Adam
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW
    if name == OptimizerName.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerName]
    raise ValueError(
        f"`{name}` is not a supported optimizer. "
        f"Supported optimizers are: {supported_optimizers}"
    )


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    if name == SchedulerName.LINEAR:
        return LinearLR
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(
        f"`{name}` is not a supported scheduler. "
        f"Supported schedulers are: {supported_schedulers}"
    )