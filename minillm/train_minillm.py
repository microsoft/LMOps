import torch
import os
import json
import torch.distributed as dist

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    ParallelOPTForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,
    ParallelLlamaForCausalLM)

parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "gptj": ParallelGPTJForCausalLM,
    "llama": ParallelLlamaForCausalLM
}

from arguments import get_args
from utils import print_args, initialize, load_parallel

from minillm import train, Reward


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    # tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"})
    if args.model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_teacher_model(args, device):
    if args.model_parallel:
        config = AutoConfig.from_pretrained(args.teacher_model_path)
        config.is_model_parallel = True
        model = parallel_model_map[args.teacher_model_type](config)
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path).to(device)
    
    if args.teacher_model_fp16:
        model = model.half()
    
    return model


def main():
    
    args = get_args()
    initialize(args)

    device = torch.cuda.current_device()
    
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
            
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    teacher_model = get_teacher_model(args, device)
    tokenizer = get_tokenizer(args)
    
    reward = Reward(args, tokenizer, teacher_model)
    
    train(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=teacher_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir,
        eval_prompt_data=args.prompt_data_dir,
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )


if __name__ == "__main__":
    main()