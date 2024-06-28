import time
import os
import sys
   
import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
import yaml
from tqdm import tqdm
import math
import copy
import time
import shutil

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    mpu,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model, parallel_model_map

from utils import plot_simple_line_chart

from accelerate import init_empty_weights

from rouge_metric import compute_metrics

from peft import PeftModel

torch.set_num_threads(4)
from tensorboardX import SummaryWriter
start_time = None

def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            if args.model_type=="qwen":
                model = parallel_model_map[args.model_type](config).to(torch.bfloat16)
            else:
                if args.dtype is not None:
                    model = parallel_model_map[args.model_type](config).to(args.dtype)
                else:
                    model = parallel_model_map[args.model_type](config).half()
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
    else:
        config.is_model_parallel = False
        # date_type = torch.bfloat16 if (args.bf16 or args.model_type=="qwen") else torch.float16
        data_type = args.dtype if args.dtype is not None else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=data_type
        )

        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.peft_path)
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        # goes here.
        if args.db_tmax:
            T_max_used = args.train_iters_per_epoch*20
        else:
            T_max_used = args.total_iters
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max_used, 
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    if args.model_type=="qwen" and ds_config['fp16']['enabled']==True:
        import copy
        ds_config['bf16']=copy.deepcopy(ds_config['fp16'])
        ds_config['fp16']['enabled']=False
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data

def find_non_repeating_lengths(data):  
    reversed_data = data.flip(dims=[1])    
    mask = reversed_data != reversed_data[:, 0:1]  
    lengths = mask.long().argmax(dim=1)  
    all_repeating = lengths == 0  
    lengths[all_repeating] = data.size(1)  
    return data.size(1) - lengths 

def get_run_time():
    global start_time
    cur_time = time.time()
    duration = cur_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    return hours, minutes, seconds


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            # dist.barrier()
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
                loss_mask = no_model_batch["loss_mask"].view(-1)
                loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step

        run_h, run_m, run_s = get_run_time()
        res['run_time'] = f'{run_h}:{run_m}:{run_s}'
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))

        return {'avg_loss': avg_loss, 'exact_match': res['exact_match'], 'rougeL': res['rougeL']}


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    random.seed(args.seed)  
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(args.seed)  
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    initialize(args)

    if dist.get_rank() == 0:
        print_args(args)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    # args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

    global start_time
    start_time = time.time()
   
    evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
