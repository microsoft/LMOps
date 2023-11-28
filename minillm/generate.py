import time
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import numpy as np

import json
from tqdm import tqdm

from transformers import mpu

from arguments import get_args

from data_utils.prompt_datasets import PromptDataset
from utils import print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import get_tokenizer, get_model


torch.set_num_threads(4)


def setup_model(args, ds_config, device):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    optimizer, lr_scheduler = None, None
        
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model


def prepare_dataset(args, tokenizer):
    data = {}
    data = PromptDataset(args, tokenizer, "train", data_path=args.data_dir, num=args.gen_num)
    print_rank("gen num", len(data))
    return data


def generate(args, tokenizer, model, dataset, device):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_gen_ids = []
    all_idxs = []
    max_new_tokens = args.max_length - args.max_prompt_length

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Generating", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            t_gen_out = model.generate(
                **model_batch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False)
    
            full_ids = t_gen_out.sequences
            gen_ids = full_ids[:, model_batch["input_ids"].size(1):]
            buffer = torch.ones(gen_ids.size(0), max_new_tokens, dtype=torch.long, device=gen_ids.device) * tokenizer.pad_token_id
            buffer[:, :gen_ids.size(1)] = gen_ids
            all_gen_ids.append(buffer)
            all_idxs.append(no_model_batch["idx"])            

    all_idxs = all_gather(torch.cat(all_idxs, dim=0), dim=0, world_size=dp_world_size, group=dp_group).cpu().tolist()
    all_gen_ids = all_gather(torch.cat(all_gen_ids, dim=0), dim=0, world_size=dp_world_size, group=dp_group).cpu().tolist()
    
    if get_rank() == 0:
        all_gen_strs = tokenizer.batch_decode(all_gen_ids, skip_special_tokens=True)
        mean_lens = np.mean([len(tokenizer.encode(x)) for x in all_gen_strs[:100]])
        
        log_str = f"gen | avg. lens: {mean_lens}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
        assert len(all_idxs) == len(all_gen_strs)

        for idx, g in zip(all_idxs, all_gen_strs):
            dataset.origin_data[idx]["gen_answer"] = g
        
        with open(os.path.join(args.save, "raw.jsonl"), "w") as f:
            for d in dataset.origin_data:
                if "gen_answer" in d:
                    f.write(json.dumps(d) + "\n")

    dist.barrier()


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"]
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    model = setup_model(args, ds_config, device)
    
    generate(args, tokenizer, model, dataset, device)


if __name__ == "__main__":
    main()
