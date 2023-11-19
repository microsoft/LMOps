from data_utils.prompt_datasets import PromptDataset
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    mpu,
    AutoConfig,)

import os
import random
import nltk
nltk.download("punkt")

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import print_rank, save_rank, load_parallel, parallel_model_map

torch.set_num_threads(4)
        

def prepare_dataset_eb(args, tokenizer):
    data = {}
    rng = random.Random(args.seed_ppo)
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def get_inputs(args, full_ids, tokenizer):
    attention_mask = (full_ids != tokenizer.pad_token_id)

    model_inputs = {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "use_cache": False
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        model_inputs["position_ids"] = position_ids
    
    return model_inputs


def calc_batch(args, tokenizer, generation_config, model, teacher_model, model_batch, gen_model_type="base"):
    query_ids = model_batch["input_ids"]
    max_new_tokens = args.max_length - query_ids.size(1)

    batch_err = []
    batch_masks = []
    if gen_model_type == "base":
        gen_model = model
    else:
        gen_model = teacher_model

    for _ in range(args.eb_sample_times):            
        gen_out = gen_model.generate(
            **model_batch,
            generation_config=generation_config,
            min_length=None,
            max_new_tokens=max_new_tokens
        )
        full_ids = gen_out.sequences
        
        full_ids = F.pad(
            full_ids,
            (0, args.max_length - full_ids.shape[1]),
            value=tokenizer.pad_token_id,
        )
    
        # response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)

        inputs = get_inputs(args, full_ids, tokenizer)
        output = model(**inputs)
        logits = output.logits
        teacher_output = teacher_model(**inputs)
        teacher_logits = teacher_output.logits
    
        logits = logits[:, query_ids.size(1):, :]
        teacher_logits = teacher_logits[:, query_ids.size(1):, :]
        masks = inputs["attention_mask"][:, query_ids.size(1):]
        
        logprobs = F.log_softmax(logits, dim=-1)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = torch.exp(teacher_logprobs)
        err1 = torch.sum(teacher_probs * (teacher_logprobs - logprobs), dim=-1)
        
        err1 = err1 * masks
        batch_err.append(err1)
        batch_masks.append(masks)

    batch_masks = torch.stack(batch_masks, dim=1)
    batch_err = torch.stack(batch_err, dim=1)
    
    mean_batch_err = torch.sum(batch_err, dim=1) / (torch.sum(batch_masks, dim=1) + 1e-3)
    mean_batch_err = torch.cumsum(mean_batch_err, dim=-1)

    return mean_batch_err


def evaluate(args, tokenizer, model, teacher_model, dataset: PromptDataset, epoch, device):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    all_R = []
    all_eps = []

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            
            mean_batch_R = calc_batch(args, tokenizer, generation_config, model, teacher_model, model_batch, gen_model_type="base")
            mean_batch_eps = calc_batch(args, tokenizer, generation_config, model, teacher_model, model_batch, gen_model_type="teacher")
            
            all_R.append(mean_batch_R)
            all_eps.append(mean_batch_eps)
            
        all_R = torch.cat(all_R, dim=0)
        mean_R = torch.mean(all_R, dim=0)
        std_R = torch.std(all_R, dim=0)
        all_eps = torch.cat(all_eps, dim=0)
        mean_eps = torch.mean(all_eps, dim=0)
        std_eps = torch.std(all_eps, dim=0)
        
        ex_acc_err = (mean_R - mean_eps) / mean_eps * 100
        std_err = (torch.abs(std_R / mean_R) + torch.abs(std_eps / std_eps)) * ex_acc_err
    
    return ex_acc_err, mean_R, mean_eps, std_err, std_R, std_eps


def get_teacher_model(args, device):
    if args.model_parallel:
        config = AutoConfig.from_pretrained(args.teacher_model_path)
        model = parallel_model_map[args.model_type](config)
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path).to(device)

    if args.teacher_model_fp16:
        model = model.half()

    return model


def evaluate_eb(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    teacher_model = get_teacher_model(args, device)
    ex_acc_err, mean_R, mean_eps, std_err, std_R, std_eps = evaluate(args, tokenizer, model, teacher_model, dataset, epoch, device)

    torch.save((ex_acc_err, mean_R, mean_eps, std_R, std_eps), os.path.join(args.save, "res.pt"))

    log_str = f"{split} | name: {args.data_names} | ExAccErr: {ex_acc_err[15]} | R: {mean_R[15]} | eps: {mean_eps[15]} | std_err: {std_err[15]} | std_R: {std_R[15]} | std_eps: {std_eps[15]}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
