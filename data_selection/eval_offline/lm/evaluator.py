import os
import time

from utils import save_rank, get_tokenizer
from train_eval_utils.base_evaluator import BaseEvaluator
from pretrain.trainer import PreTrainer


class LMEvaluator(BaseEvaluator):
    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device)
        
    def setup(self):
        super().setup()
        self.base_model_path = self.args.model_path
        self.base_output_path = self.args.save
        # modify args for trainer
        self.args.model_path = self.model_path
        self.args.save = self.output_path
        self._trainer = PreTrainer(self.args, self.ds_config, self.device, do_train=False)
        
    def before_eval_step_callback(self):
        self.args.model_path = os.path.join(self.base_model_path, f"{self.global_steps}")
        self.args.save = os.path.join(self.base_output_path, f"{self.global_steps}")
        self.tokenizer = get_tokenizer(self.args, model_path=self.args.model_path)
        os.makedirs(self.args.save, exist_ok=True)
        self._trainer.setup_model_and_optimizer(self.args, set_optim=False)
    
    def _evaluate(self):
        self.print_and_save(f"Evaluating {self.args.model_path}", self.base_output_path)
        self.print_and_save(f"Results will be saved to {self.args.save}", self.base_output_path)
        time_tag = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        save_rank(f"eval | {time_tag} | {self.args.data_name}", os.path.join(self.args.save, "log.txt"))
        
        self._trainer.evaluate()
        
        return None
