import torch
import torch.distributed as dist
import random
import numpy as np
import os
from datetime import timedelta


def get_distribution(logits, temperature):
    probs = torch.softmax(logits.to(torch.float32) / (temperature + 1e-10), dim=-1, dtype=torch.float32)
    return probs


def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)


def sample_from_draft_model(model, initial_prompt_seq, new_tokens, eos_token_id, temperature=1.0):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
        out_logits.append(sample_token_logits)
        if sample_token == eos_token_id:
            break        

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits


# Logging
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


# Distributed
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


# Initialize
def set_random_seed(seed):
    """Set random seed for reproducability."""
    if dist.is_initialized():
        seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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

def initialize(args, do_distributed=True):
    # init distributed
    if do_distributed:
        init_distributed(args)

    set_random_seed(args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
