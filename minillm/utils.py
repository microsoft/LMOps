from collections import defaultdict
from typing import Dict
import numpy as np
import os
import torch.distributed as dist
from torch.distributed import get_rank
import torch.nn as nn
import random
import torch
from datetime import timedelta
import deepspeed
import transformers.mpu as mpu


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if mp:
            mpu.model_parallel_cuda_manual_seed(seed)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    if args.model_parallel:
        assert dist.get_world_size() % args.model_parallel_size == 0 
        mpu.initialize_model_parallel(args.model_parallel_size)

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)


def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def load_parallel(model, load_dir):
    mp_rank = mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_world_size() != 1
    checkpoint_name = os.path.join(load_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    assert os.path.exists(checkpoint_name), f"{checkpoint_name} does not exist."
    sd = torch.load(checkpoint_name, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    dist.barrier()
    print(f"Rank {get_rank()}: {checkpoint_name} loaded.")


def save_parallel(model, save_dir):
    mp_rank = mpu.get_model_parallel_rank()
    os.makedirs(os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}"), exist_ok=True)
    checkpoint_name = os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Rank {get_rank()}: {checkpoint_name} saved.")
