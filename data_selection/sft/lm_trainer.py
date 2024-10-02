import os
from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, save_rank
from torch.distributed import get_rank
from data_utils.lm_datasets import LMDataset
import wandb


class SFTLMTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
    
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            if args.dev_data_dir is None:
                args.dev_data_dir = args.data_dir
            self.train_dataset = LMDataset(args, self.tokenizer, "train", args.data_dir, args.train_num, ada_max_length=True)
            print_rank("### Training Data Number:", len(self.train_dataset))
            self.eval_dataset = LMDataset(args, self.tokenizer, "dev", args.dev_data_dir, args.dev_num, ada_max_length=True)
            print_rank("### Dev Data Number:", len(self.eval_dataset))
        else:
            self.eval_dataset = LMDataset(args, self.tokenizer, "test", args.data_dir, args.dev_num, ada_max_length=True)
    
    def compute_loss(self, model_batch, no_model_batch):
        return self.compute_lm_loss(model_batch, no_model_batch), {}

    def evaluate(self):
        lm_res = self.evaluate_lm()

        if get_rank() == 0:
            res = {
                "lm_loss": lm_res["avg_loss"],
            }
                        
            wandb.log(res, step=self.global_steps)
            
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)

