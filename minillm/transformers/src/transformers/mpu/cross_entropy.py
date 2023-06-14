# coding=utf-8



import torch
import torch.distributed as dist

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


class _ParallelSoftmax(torch.autograd.Function):
    
    @staticmethod
    def forward(cls, x_in: torch.Tensor, dim=-1) -> torch.Tensor:
        assert dim == -1
        # NOTE: dim=-1
        # Copy so the input remains unchanged.
        x = x_in.clone()
        # Maximum value along vocab dimension across all GPUs.
        x_max = torch.max(x, dim=-1)[0]
        dist.all_reduce(x_max,
                        op=dist.ReduceOp.MAX,
                        group=get_model_parallel_group())

        # Subtract the maximum value.
        x.sub_(x_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_x = x.exp()
        sum_exp_x = exp_x.sum(dim=-1)
        dist.all_reduce(sum_exp_x,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        softmax = torch.div(exp_x, sum_exp_x.unsqueeze(dim=-1))
        cls.save_for_backward(softmax)

        return softmax

    @staticmethod
    def backward(cls, grad_output):
        # raise NotImplementedError # not tested
        softmax, = cls.saved_tensors
        
        x = torch.sum(grad_output * softmax, dim=-1)
        dist.all_reduce(x,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())
        
        grad_input = softmax * grad_output - x.unsqueeze(-1) * softmax

        return grad_input, None


class _ParallelLogSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(cls, x_in, dim=-1):
        assert dim == -1
        # NOTE: dim=-1
        # Copy so the input remains unchanged.
        x = x_in.clone()
        # Maximum value along vocab dimension across all GPUs.
        x_max = torch.max(x, dim=-1)[0]
        dist.all_reduce(x_max,
                        op=dist.ReduceOp.MAX,
                        group=get_model_parallel_group())

        # Subtract the maximum value.
        x.sub_(x_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_x = x.exp()
        sum_exp_x = exp_x.sum(dim=-1)
        dist.all_reduce(sum_exp_x,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())
        
        log_sum_exp_x = torch.log(sum_exp_x)

        log_softmax = x - log_sum_exp_x.unsqueeze(dim=-1)

        cls.save_for_backward(exp_x, sum_exp_x)

        return log_softmax

    @staticmethod
    def backward(cls, grad_output):
        # raise NotImplementedError # not tested
        exp_x, sum_exp_x = cls.saved_tensors
        softmax = torch.div(exp_x, sum_exp_x.unsqueeze(dim=-1))
        grad_out_sum = torch.sum(grad_output, dim=-1)
        dist.all_reduce(grad_out_sum,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())
        grad_input = grad_output - grad_out_sum.unsqueeze(-1) * softmax
        
        return grad_input, None


class _ParallelSoftCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        dist.all_reduce(logits_max,
                        op=dist.ReduceOp.MAX,
                        group=get_model_parallel_group())
        # Subtract the maximum value.
        logits = logits - logits_max.unsqueeze(dim=-1)
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        targets_max = torch.max(targets, dim=-1)[0]
        dist.all_reduce(targets_max,
                        op=dist.ReduceOp.MAX,
                        group=get_model_parallel_group())
        # Subtract the maximum value.
        targets = targets - targets_max.unsqueeze(dim=-1)
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_targets = targets.exp()
        sum_exp_targets = exp_targets.sum(dim=-1)
        dist.all_reduce(sum_exp_targets,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        # targets_softmax: [b, s, v_p]
        targets_softmax = torch.div(exp_targets, sum_exp_targets.unsqueeze(-1))

        # sum_targets_softmax_logits: [b, s]
        sum_targets_softmax_logits = torch.matmul(
            targets_softmax.unsqueeze(-2), logits.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        dist.all_reduce(sum_targets_softmax_logits,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        loss = torch.log(sum_exp_logits) - sum_targets_softmax_logits #+ sum_log_targets_softmax

        logits_softmax = torch.div(exp_logits, sum_exp_logits.unsqueeze(-1))

        cls.save_for_backward(logits_softmax, targets_softmax)

        return loss

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_softmax, targets_softmax = cls.saved_tensors
        grad_input = (logits_softmax - targets_softmax) * grad_output.unsqueeze(-1)

        return grad_input, None


class _ParallelLogProbs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        # Copy so the input remains unchanged.
        logits = vocab_parallel_logits.clone()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        dist.all_reduce(logits_max,
                        op=dist.ReduceOp.MAX,
                        group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


class _ParallelGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, dim, ids):
        partition_size = logits.size(dim)
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        get_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        start_index, end_index = get_range(partition_size, rank, world_size)
        ids_mask = (ids < start_index) | (ids >= end_index)
        masked_ids = ids - start_index
        masked_ids[ids_mask] = 0
        gathered_logits = torch.gather(logits, dim, masked_ids)
        gathered_logits[ids_mask] = 0
        dist.all_reduce(gathered_logits,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())
        
        ctx.save_for_backward(ids_mask, masked_ids, torch.tensor([partition_size, dim], device=logits.device))
        
        return gathered_logits

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError # not tested
        # Retreive tensors from the forward path.
        ids_mask, masked_ids, ints = ctx.saved_tensors
        partition_size, dim = ints
        
        size = ids_mask.size()[:-1] + (partition_size,)
        
        grad_input = torch.zeros(size, dtype=grad_output.dtype, device=grad_output.device)
        
        grad_input.scatter_(dim, masked_ids, (1 - ids_mask.to(grad_output.dtype)))

        return grad_input, None, None


class _ParallelSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, dim):
        assert dim == -1
        logits_sumed = torch.sum(logits, dim=-1)
        dist.all_reduce(logits_sumed,
                        op=dist.ReduceOp.SUM,
                        group=get_model_parallel_group())
        ctx.save_for_backward(torch.tensor([logits.size(-1)], device=logits.device))
        
        return logits_sumed

    @staticmethod
    def backward(ctx, grad_output):
        dim_size, = ctx.saved_tensors
        dim_size = dim_size.item()
        repeat_size = [1 for _ in range(grad_output.dim())] + [dim_size]
        grad_input = grad_output.unsqueeze(-1).repeat(repeat_size)

        return grad_input, None



def parallel_softmax(logits, dim):
    return _ParallelSoftmax.apply(logits, dim)


def parallel_log_softmax(logits, dim):
    return _ParallelLogSoftmax.apply(logits, dim)


def parallel_soft_cross_entropy_loss(logits, targets):
    return _ParallelSoftCrossEntropyLoss.apply(logits, targets)


def parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _ParallelLogProbs.apply(vocab_parallel_logits, target)


def parallel_logprobs(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _ParallelLogProbs.apply(vocab_parallel_logits, target)


def parallel_gather(logits, dim, ids):
    return _ParallelGather.apply(logits, dim, ids)


def parallel_logsumexp(logits, dim=-1):
    # NOTE: dim is the model parallel dim
    sum_exp_x = torch.sum(torch.exp(logits), dim=dim)
    dist.all_reduce(sum_exp_x,
                    op=dist.ReduceOp.SUM,
                    group=get_model_parallel_group())
    log_sum_exp_x = torch.log(sum_exp_x)
    return log_sum_exp_x


def parallel_sum(x, dim=-1):
    # NOTE: dim is the model parallel dim
    # x = torch.sum(x, dim=dim)
    # dist.all_reduce(x,
    #                 op=dist.ReduceOp.SUM,
    #                 group=get_model_parallel_group())
    x = _ParallelSum.apply(x, dim)
    return x


def parallel_mean(x, dim=-1):
    # NOTE: dim is the model parallel dim
    dim_size = x.size(dim)
    x = torch.sum(x, dim=dim)
    dist.all_reduce(x,
                    op=dist.ReduceOp.SUM,
                    group=get_model_parallel_group())
    full_dim = dim_size * get_model_parallel_world_size()
    x = x / full_dim
    return x
    