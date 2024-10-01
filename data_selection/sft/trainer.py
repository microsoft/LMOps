import os
from train_eval_utils.base_trainer import BaseTrainer
from utils import print_rank, save_rank
from torch.distributed import get_rank

class SFTTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
    
    def compute_loss(self, model_batch, no_model_batch):
        return self.compute_lm_loss(model_batch, no_model_batch), {}

    def evaluate(self):
        if self.args.eval_ppl:
            lm_res = self.evaluate_lm()
        else:
            lm_res = {}
        if self.args.eval_gen:
            prompt_ids, response_ids, gen_res, response_strs = self.evaluate_gen()
        else:
            gen_res = {}
            response_strs = []
            prompt_ids = []
            response_ids = []

        if get_rank() == 0:
            res = {**lm_res, **gen_res}

            if self.args.eval_gen:
                # for i in range(3):
                #     print_rank(f"Input:\n{self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)}\n")
                #     print_rank(f"Output:\n{response_strs[i]}\n")
                #     print_rank(f"Ground Truth:\n{self.eval_dataset.answers[i]}\n")
                #     print_rank("*" * 100)

                self.save_evals(response_ids, res, response_strs)
            
            eval_log_str = self.get_log(res, "eval")
            print_rank("*" * 100)
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))
            print_rank("*" * 100)
