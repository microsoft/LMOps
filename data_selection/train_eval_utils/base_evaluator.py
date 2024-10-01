import os
import uuid
import wandb
from time import time
import torch.distributed as dist

from utils import print_rank, save_rank
from utils import WANDB_PROJ_NAME, get_tokenizer

try:
    from transformers import mpu
except ImportError:
    mpu = None


class BaseEvaluator():
    def __init__(self, args, ds_config, device):
        self.args = args
        self.ds_config = ds_config
        self.device = device
        self.global_steps = 0
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.wandb_name = self.args.wandb_name if self.args.wandb_name is not None else self.exp_name
        self.group_name = self.args.wandb_group or "pad"

        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None

        self.start = self.args.eval_start_ckpt
        self.end = self.args.eval_end_ckpt
        self.interval = self.args.eval_interval
        
        self.setup()

    def setup(self):
        self.print_and_save(f"Model path: {self.args.model_path}")
        self.print_and_save(f"Data name {self.args.data_name}")
        if self.args.eval_start_ckpt is not None and self.args.eval_end_ckpt is not None:
            self.print_and_save(f"Start ckpt: {self.args.eval_start_ckpt} | End ckpt: {self.args.eval_end_ckpt} | Interval: {self.args.eval_interval}")
            self.model_path = os.path.join(self.args.model_path, f"{self.args.eval_start_ckpt}")
            self.output_path = os.path.join(self.args.save, f"{self.args.eval_start_ckpt}")
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.model_path = self.args.model_path
            self.output_path = self.args.save
        
        self.tokenizer = get_tokenizer(self.args, self.model_path)

    def before_eval_step_callback(self):
        self.model_path = os.path.join(self.args.model_path, f"{self.global_steps}")
        self.output_path = os.path.join(self.args.save, f"{self.global_steps}")
        self.tokenizer = get_tokenizer(self.args, model_path=self.model_path)
        os.makedirs(self.output_path, exist_ok=True)
    
    def after_eval_step_callback(self, results):
        pass

    def _evaluate(self):
        raise NotImplementedError

    def evaluate(self):
        if self.dp_rank == 0:
            wandb_id = self.args.wandb_id or (str(int(time())) + "-" + str(uuid.uuid4()))
            run = wandb.init(
                id=wandb_id,
                name=self.wandb_name,
                project=WANDB_PROJ_NAME,
                group=self.group_name,
                config=self.args,
                reinit=True,
                tags=[self.args.time_stamp],
                mode=self.args.wandb_mode)
        
        if self.start is None or self.end is None:
            results = self._evaluate()
        else:
            assert self.interval > 0
            for s in range(self.start, self.end + 1, self.interval):
                self.global_steps = s
                self.before_eval_step_callback()
                results = self._evaluate()
                self.after_eval_step_callback(results)
        
        if self.dp_rank == 0:
            run.finish()
                
    def print_and_save(self, log_str, output_path=None):
        output_path = output_path or self.args.save
        print_rank(log_str)
        save_rank(log_str, os.path.join(output_path, "log.txt"))
