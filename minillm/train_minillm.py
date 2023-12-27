import torch
import os
import json
import torch.distributed as dist
from accelerate import init_empty_weights

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,)

from arguments import get_args
from utils import print_args, initialize, load_parallel, get_tokenizer, parallel_model_map

from minillm import train, Reward

from peft import PeftModel


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            if args.model_type=="qwen":
                model = parallel_model_map[args.model_type](config).to(torch.bfloat16)
            else:
                model = parallel_model_map[args.model_type](config).half()
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=torch.float16 if args.model_type!="qwen" else torch.bfloat16
        )

        if args.peft is not None:
            if args.peft == "lora":
                assert args.teacher_peft_path is not None
                model = PeftModel.from_pretrained(model, args.peft_path)
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()

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
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    args.deepspeed_config = None
    
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