from typing import Dict
import numpy as np
import os
import time
import torch.distributed as dist
from torch.distributed import get_rank
import random
import torch
import torch.nn as nn
from datetime import timedelta
import deepspeed
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    ParallelOPTForCausalLM,
    ParallelLlamaForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,
    ParallelMistralForCausalLM,
    ParallelQWenLMHeadModel,
    mpu,)


parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gptj": ParallelGPTJForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "llama": ParallelLlamaForCausalLM,
    "llama2": ParallelLlamaForCausalLM,
    "mistral": ParallelMistralForCausalLM,
    "qwen": ParallelQWenLMHeadModel,
}


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


# Load and save model
def get_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path)
    
    st_time = time.time()
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            if args.model_type=="qwen":
                model = parallel_model_map[args.model_type](config).to(torch.bfloat16)
            else:
                model = parallel_model_map[args.model_type](config).half()
        load_parallel(model, args.model_path)

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)
    else:
        config.is_model_parallel = False
        if args.model_type=="qwen":
            dtype = torch.float32 if args.fp32 else torch.float16
        else:
            dtype = torch.float32 if args.fp32 else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map={"": device}, torch_dtype=dtype)

        if args.peft is not None:
            if args.peft == "lora":
                model.enable_input_require_grads()
                if args.peft_path is not None:
                    model = PeftModel.from_pretrained(model, args.peft_path)
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, inference_mode=(not args.do_train), r=args.peft_lora_r, lora_alpha=args.peft_lora_alpha, lora_dropout=args.peft_lora_dropout
                    )
                    model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)
        # model = DDP(model)
        # NOTE: no need for DDP since deepspeed has done
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    print_rank(f"Model load time: {ed_time - st_time}s")
    
    return model


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


def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_type=="qwen":
        tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def load_parallel(model, load_dir):
    mp_rank = mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_world_size() != 1
    checkpoint_name = os.path.join(load_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    assert os.path.exists(checkpoint_name), f"{checkpoint_name} does not exist."
    model = load_checkpoint_and_dispatch(model=model, checkpoint=checkpoint_name, device_map={"": torch.cuda.current_device()}, dtype=torch.float16)
    dist.barrier()
    print(f"Rank {get_rank()}: {checkpoint_name} loaded.")


def save_parallel(model, save_dir):
    mp_rank = mpu.get_model_parallel_rank()
    os.makedirs(os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}"), exist_ok=True)
    checkpoint_name = os.path.join(save_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Rank {get_rank()}: {checkpoint_name} saved.")
