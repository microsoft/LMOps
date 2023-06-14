from data_utils.prompt_datasets import PromptDataset
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    mpu,
    AutoConfig,
    ParallelOPTForCausalLM,
    ParallelLlamaForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,)

parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gptj": ParallelGPTJForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "llama": ParallelLlamaForCausalLM
}

import os
import random
import nltk
nltk.download("punkt")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, load_parallel

from rouge_metric import compute_metrics

torch.set_num_threads(4)


def prepare_dataset_main(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def run_model(args, tokenizer, model, dataset: PromptDataset, epoch, device):
    
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
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []
    
    generation_config = GenerationConfig (
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

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss)

            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            all_query_ids.extend(query_ids)
            all_response_ids.extend(response_ids)

    all_lm_losses = torch.cat(all_lm_losses)
    mean_lm_loss = all_lm_losses.mean().item()
        
    return (
        mean_lm_loss,
        all_query_ids,
        all_response_ids)


def get_reward_model(args, device):
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


def evaluate_main(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
        
    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:    
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {gen_res} | lm_loss {round(lm_loss, 4)} | avg. gen lenth: {mean_gen_length}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
