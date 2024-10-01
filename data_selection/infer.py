import time
import os

import torch
import torch.distributed as dist

import json
from arguments import get_args

from utils import print_args, initialize, save_rank

from data_scorer import DataScorerTrainer

torch.set_num_threads(16)


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)        
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)

        ds_config["zero_optimization"]["stage"] = 0
        
        if not ds_config["fp16"]["enabled"]:
            args.fp32 = True
        
        args.deepspeed_config = None
    else:
        ds_config = None
    
    if args.type == "data_scorer":
        trainer = DataScorerTrainer(args, ds_config, device, args.do_train)
    else:
        raise ValueError(f"Invalid type: {args.type}")    
    
    trainer.inference()

    
if __name__ == "__main__":
    main()
