import os
import re
import wandb
from collections import defaultdict
from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, save_rank
from torch.distributed import get_rank
from data_utils.prompt_datasets import PromptDataset
from data_utils.lm_datasets import LMDataset


class PreTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
        if do_train and self.args.resume_training:
            self.resume_training()
        elif args.start_from_global_step is not None:
            self.last_global_steps = self.args.start_from_global_step

    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        data_split = args.data_split or "data"
        if do_train:
            if args.dev_data_dir is None or os.path.samefile(args.dev_data_dir, args.data_dir):
                print_rank("### Spliting dev data from training data ###")
                args.dev_data_dir = args.data_dir
                min_train_offset = 100000
                assert 0 == 1, "dangerous!"
            else:
                min_train_offset = 0
            self.train_dataset = LMDataset(args, self.tokenizer, data_split, args.data_dir, args.train_num, data_name="lm", min_offset=min_train_offset+self.args.min_offset, min_state=self.args.min_state)
            self.print_and_save(f"### Training Data Number: {len(self.train_dataset)}")
            # self.train_dataset = LMDataset(args, self.tokenizer, "data", args.data_dir, args.dev_num, max_offset=10000)
            # print_rank("train num", len(self.train_dataset))
            self.eval_dataset = LMDataset(args, self.tokenizer, data_split, args.dev_data_dir, args.dev_num, data_name="lm_dev", max_offset=100000)
            self.print_and_save(f"### Dev Data Number: {len(self.eval_dataset)}")
        else:
            self.eval_dataset = LMDataset(args, self.tokenizer, data_split, args.data_dir, args.test_num, max_offset=100000)
            self.print_and_save(f"### Test Data Number: {len(self.eval_dataset)}")
    
    def compute_loss(self, model_batch, no_model_batch):
        return self.compute_lm_loss(model_batch, no_model_batch), {}

    def evaluate(self):
        lm_res = self.evaluate_lm()

        if get_rank() == 0:
            res = {
                "lm_loss": lm_res["avg_loss"],
            }
                        
            if self.do_train and wandb.run is not None:
                wandb.log(res, step=self.global_steps)
            
            eval_log_str = self.get_log(res, "eval")
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
