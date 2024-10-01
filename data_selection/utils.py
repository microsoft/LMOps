import torch
import deepspeed
import torch.distributed as dist
import random
import numpy as np
import os
from datetime import timedelta
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import time
from torch.distributed import get_rank
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

try:
    from transformers import mpu
except:
    mpu = None

WANDB_PROJ_NAME = "data_selection"
PAD_EOS_MODELS = ["gpt2", "opt", "llama", "mistral"]
BOS_MODELS = ["fairseq", "mistral", "llama"]


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
    if dist.is_initialized():
        seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if mp:
        #     mpu.model_parallel_cuda_manual_seed(seed)


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


def initialize(args, do_distributed=True):
    # init distributed
    if do_distributed:
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
def get_model(args, device, model_path=None, config=None, from_scratch=None, model_cls=None):
    if model_path is None:
        model_path = args.model_path
    print_rank("Initializing model from {}".format(model_path), rank=0)
    print_rank(f"Attention Implementation: {args.attn_impl}")
    if config is None:
        config = AutoConfig.from_pretrained(model_path, attn_implementation=args.attn_impl)
        
    if args.dropout_path_rate is not None:
        config.drop_path_rate = args.dropout_path_rate
    if args.xops_attn:
        assert args.attn_impl == "eager"
        print_rank("Xops Attention")
        config.use_memory_efficient_attention = True
    else:
        config.use_memory_efficient_attention = False

    st_time = time.time()
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = parallel_model_map[args.model_type].half()
        load_parallel(model, args.model_path)

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)
    else:
        config.is_model_parallel = False
        from_scratch = from_scratch if from_scratch is not None else args.from_scratch
        model_cls = model_cls if model_cls is not None else AutoModelForCausalLM
        if from_scratch:
            model = model_cls.from_config(config, attn_implementation=args.attn_impl).to(device)
        else:
            dtype = torch.float32 if args.fp32 else torch.float16
            model = model_cls.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=dtype)

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


def get_tokenizer(args, model_path=None, model_type=None):
    if model_path is None:
        model_path = args.model_path
    
    if model_type is None:
        model_type = args.model_type

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_type in PAD_EOS_MODELS:
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
    
    
def copy_to_blob(base_path, source_dir, target_dir, rm_source=False):
    target_dir = target_dir.strip("/")
    with open(os.path.join(base_path, "downstream_data", "sas_token"), "r") as f:
        sas_token = f.read().strip()
    cmd = f"{base_path}/azcopy copy {source_dir} --recursive=true \"https://msranlpintern.blob.core.windows.net/yuxian/sps/{target_dir}{sas_token}\""
    print(cmd)
    os.system(cmd)
    if rm_source:
        print(f"rm -rf {source_dir}")
        os.system(f"rm -rf {source_dir}")


def copy_from_blob(base_path, source_dir, target_dir, rm_source=False):
    source_dir = source_dir.strip("/")
    with open(os.path.join(base_path, "downstream_data", "sas_token"), "r") as f:
        sas_token = f.read().strip()
    cmd = f"{base_path}/azcopy copy --recursive=true \"https://msranlpintern.blob.core.windows.net/yuxian/sps/{source_dir}{sas_token}\" {target_dir}"
    print(cmd)
    os.system(cmd)


def naive_copy_to_blob(base_path, source_dir, target_dir, rm_source=False):
    cmd = f"cp -r {source_dir} {base_path}/{target_dir}"
    print(cmd)
    os.system(cmd)
    if rm_source:
        print(f"rm -rf {source_dir}")
        os.system(f"rm -rf {source_dir}")


def remove_path(path):
    print("Remove", path)
    if os.path.exists(path):
        os.system(f"rm -rf {path}")