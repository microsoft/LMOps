import os
import time
import json
import torch

from perceptron_trainer import PerceptronTrainer
from transformer_trainer import TransformerTrainer
from opt_policy_trainer import OptPolicyTrainer
from eval_policy_trainer import EvalPolicyTrainer
from arguments import get_args
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import print_args, initialize, save_rank

torch.backends.cudnn.enabled = False


def main():
    args = get_args()
    initialize(args, do_distributed=True)
    mp.set_start_method("spawn")
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    args.time_stamp = cur_time
    
    if args.opt_gamma:
        trainer_cls = OptPolicyTrainer 
    elif args.eval_opt_gamma:
        trainer_cls = EvalPolicyTrainer
    else:
        if args.data_names == "tinystory":
            trainer_cls = TransformerTrainer
        elif args.data_names == "linear":
            trainer_cls = PerceptronTrainer
        else:
            raise NotImplementedError
    
    trainer = trainer_cls(args, device)
    if args.wandb_name is not None:
        trainer.train(wandb_name=args.wandb_name)
    else:
        trainer.train()
    
if __name__ == "__main__":
    main()