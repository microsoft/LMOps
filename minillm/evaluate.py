import time
import os

import torch
import torch.distributed as dist
import deepspeed

import json

from transformers import mpu

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model

from evaluate_main import evaluate_main, prepare_dataset_main
from evaluate_exposure_bias import evaluate_eb, prepare_dataset_eb


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
    print("OK")
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"] 
    args.deepspeed_config = None

    # get the tokenizer
    tokenizer = get_tokenizer(args)
    if args.type == "eval_main":
        dataset = prepare_dataset_main(
            args,
            tokenizer,
        )
    elif args.type == "eval_exposure_bias":
        dataset = prepare_dataset_eb(
            args,
            tokenizer,
        )
    else:
        raise NotImplementedError
    model = setup_model(args, ds_config, device)
    
    if args.type == "eval_main":
        evaluate_main(args, tokenizer, model, dataset["test"], "test", 0, device)
    elif args.type == "eval_exposure_bias":
        evaluate_eb(args, tokenizer, model, dataset["test"], "test", 0, device)
    else:
        raise NotImplementedError
    
    
if __name__ == "__main__":
    main()