import os
import re
import uuid
import json
import math
import wandb
import random
import deepspeed
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW, SGD, Adam
from data_utils.prompt_datasets import PromptDataset

from transformers import (
    GenerationConfig,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from utils import print_rank, save_rank, save_parallel, all_gather
from utils import get_model, get_tokenizer
from utils import WANDB_PROJ_NAME

from .schedulers import WarmupCosineAnnealingLR

try:
    from transformers import mpu
except ImportError:
    mpu = None


class BaseTrainer():
    def __init__(self, args, ds_config, device, do_train=True):
        self.args = args
        self.ds_config = ds_config
        self.device = device
        self.do_train = do_train
        self.tokenizer = get_tokenizer(args)
        self.grad_norm = 0
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        self.wandb_name = self.args.wandb_name if self.args.wandb_name is not None else self.exp_name
        self.group_name = self.args.wandb_group or "pad"
        self.global_steps = None
        self.steps = None
        self.epoch = None
        self.epochs = None
        self.total_steps = None
        self.first_printed = False
        
        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None
        
        # if self.args.torch_compile is not None:
        #     self._train_pass = torch.compile(self._train_pass, mode=self.args.torch_compile)
    
    def get_model(self, args=None, device=None):
        args = args or self.args
        device = device or self.device
        return get_model(args, device)
    
    def get_optimizer(self, model, args=None):
        args = args or self.args
        if self.args.optimizer_name.lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif self.args.optimizer_name.lower() == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps, betas=(args.adam_beta, args.adam_beta2))
        elif self.args.optimizer_name.lower() == "adamw":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps, betas=(args.adam_beta, args.adam_beta2))
        else:
            raise ValueError(f"Optimizer of type {self.args.optimizer_name} is not supported yet.")
        print_rank(f'Optimizer = {optimizer.__class__.__name__}')
        return optimizer
        
    def get_lr_scheduler(self, optimizer, args=None):
        args = args or self.args
        assert self.total_steps is not None and self.total_steps > 0
        if args.scheduler_name == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters)
        elif args.scheduler_name == "cosine":
            lr_scheduler = WarmupCosineAnnealingLR(
                optimizer,
                T_max=self.total_steps,
                warmup_steps=args.warmup_iters,
                eta_min=args.lr_min)
        elif args.scheduler_name == "noam":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_iters,
                num_training_steps=self.total_steps,
                power=0.5)
        else:
            raise ValueError(f"lr_scheduler of type {args.scheduler_name} is not supported yet.")

        return lr_scheduler
    
    def setup_model_and_optimizer(self, args=None, ds_config=None, device=None, set_optim=True):
        args = args or self.args
        device = device or self.device
        ds_config = ds_config or self.ds_config
        # get the model
        model = self.get_model(args, device)
        # get the optimizer and lr_scheduler
        if set_optim:
            optimizer = self.get_optimizer(model, args)
            lr_scheduler = self.get_lr_scheduler(optimizer, args)
        else:
            optimizer, lr_scheduler = None, None
            
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.model_parallel else None,
            config_params=ds_config
        )
        
        # get the memory usage
        print_rank("Model mem\n", torch.cuda.memory_summary())
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        if self.args.torch_compile is not None:
            print_rank(f"Torch Compile Mode: {self.args.torch_compile}")
            self.model = torch.compile(self.model, mode=self.args.torch_compile)

    def resume_training(self):
        load_dir = self.args.resume_dir or self.args.save
        if self.args.resume_tag is None:
            with open(os.path.join(load_dir, "latest")) as f:
                tag = f.read().strip()
        else:
            tag = self.args.resume_tag
        self.model.load_checkpoint(load_dir, tag=tag)
        self.last_rng_states = torch.load(os.path.join(load_dir, tag, f"rng_states_{self.dp_rank}.pt"))
        
        with open(os.path.join(load_dir, tag, "dynamics.json"), "r") as f:
            dynamics = json.load(f)
        self.last_steps = dynamics["step"]
        self.last_epochs = dynamics["epoch"]
        self.last_global_steps = dynamics["global_steps"]
        self.train_dataset.set_skip_offset(dynamics["skip_offset"])
        
        print_rank(f"Resume from {load_dir} {tag}")
        print_rank(f"Resume from step {self.last_steps}, epoch {self.last_epochs}, global step {self.last_global_steps}")
 
    def prepare_learning(self, args=None):
        args = args or self.args
        self.total_batch_size = args.batch_size * self.dp_world_size * args.gradient_accumulation_steps
        self.train_iters_per_epoch = int(len(self.train_dataset) / self.total_batch_size)
        assert (args.epochs is not None) ^ (args.total_iters is not None), (args.epochs, args.total_iters)
        self.total_steps = args.total_iters or self.train_iters_per_epoch * args.epochs
        self.epochs = args.epochs or math.ceil(args.total_iters / self.train_iters_per_epoch)
        self.train_dataset.set_num(self.train_iters_per_epoch * self.total_batch_size) # droplast
        
        if args.save_interval == -1:
            args.save_interval = self.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = self.train_iters_per_epoch

        if self.args.precompute_data_order and (not self.args.resume_training):
            if self.dp_rank == 0:
                normal_order = np.arange(0, len(self.train_dataset), dtype=np.int32)
                order = np.stack([np.random.permutation(normal_order) for _ in range(self.epochs)], axis=0)
                order = order[:, :self.train_iters_per_epoch * self.total_batch_size] # droplast
                np.save(os.path.join(self.args.save, "data_order.npy"), order)
                print("Data order size: ", order.shape)
            dist.barrier()
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))
                        
        if self.args.resume_training:
            assert self.args.precompute_data_order
            assert os.path.exists(os.path.join(self.args.save, "data_order.npy"))
            self.train_dataset.set_order(path=os.path.join(self.args.save, "data_order.npy"))

        self.print_and_save(f"Total batch size: {self.total_batch_size}")
        self.print_and_save(f"Total iters: {self.total_steps}")
        self.print_and_save(f"Total epochs: {self.epochs}")
        self.print_and_save(f"Train iters per epoch: {self.train_iters_per_epoch}")
        self.print_and_save(f"Save interval: {args.save_interval}")
        self.print_and_save(f"Eval interval: {args.eval_interval}")
    
    def prepare_inference(self, args=None):
        raise NotImplementedError
     
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = PromptDataset(args, self.tokenizer, "train", args.data_dir, args.train_num, ada_max_length=True)
            print_rank("train num", len(self.train_dataset))
            self.eval_dataset = PromptDataset(args, self.tokenizer, "dev", args.data_dir, args.dev_num, ada_max_length=True)
        else:
            data_split = args.data_split or "test"
            self.eval_dataset = PromptDataset(args, self.tokenizer, data_split, args.data_dir, args.dev_num, ada_max_length=True)

    def compute_loss(self, model_batch, no_model_batch):
        raise NotImplementedError

    def _get_lm_loss_from_logits(self, logits, label, loss_mask):        
        if self.args.model_parallel:
            loss_func = mpu.parallel_cross_entropy
            lm_losses = loss_func(logits.contiguous().float(), label)
        else:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_losses = loss_func(logits.float().view(-1, logits.shape[-1]), label.view(-1))
            lm_losses = lm_losses.view(-1, label.size(-1))
        lm_loss = torch.sum((lm_losses * loss_mask), dim=-1) / torch.sum(loss_mask, dim=-1)
        return lm_loss

    def compute_lm_loss(self, model_batch, no_model_batch, mean=True):        
        outputs = self.model(**model_batch, use_cache=False)
        logits = outputs.logits

        lm_loss = self._get_lm_loss_from_logits(logits, no_model_batch["label"], no_model_batch["loss_mask"])
        
        if mean:
            lm_loss = lm_loss.mean()            
        
        return lm_loss

    def print_and_save(self, log_str, output_path=None):
        output_path = output_path or self.args.save
        print_rank(log_str)
        save_rank(log_str, os.path.join(output_path, "log.txt"))

    def get_log(self, stats, phase, **kwargs):
        log_prefix = "{} | epoch {}/{} | steps {} | global_steps {}/{}".format(
            phase,
            self.epoch,
            self.epochs,
            self.steps,
            self.global_steps,
            self.total_steps
        )
        
        log_midfix = " | ".join([f"{k}: {v:.4f}" for k,v in stats.items()])
        log_suffix = " | ".join([(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}") for k,v in kwargs.items()])
        
        return log_prefix + " | " + log_midfix + " | " + log_suffix

    def backward(self, loss, loss_stats=None):
        self.model.backward(loss)

    def _all_reduce_loss(self, loss):
        dist.all_reduce(loss, group=self.dp_group, op=dist.ReduceOp.SUM)
        return (loss / self.dp_world_size).item()

    def first_print(self, model_batch, no_model_batch, save_name=""):
        if self.dp_rank == 0:
            print(model_batch["input_ids"][0].cpu().tolist())
            print(self.tokenizer.decode(model_batch["input_ids"][0].cpu().tolist(), skip_special_tokens=True))
            print(model_batch["attention_mask"][0].cpu().tolist())
            print(no_model_batch["label"][0].cpu().tolist())
            print(no_model_batch["loss_mask"][0].int().cpu().tolist())
            torch.save(model_batch, os.path.join(self.args.save, f"model_batch_{save_name}_0.pt"))
            torch.save(no_model_batch, os.path.join(self.args.save, f"no_model_batch_{save_name}_0.pt"))

    def set_train(self):
        self.model.train()

    def _train_pass(self, model_batch, no_model_batch, stats):
        # forward
        torch.cuda.synchronize()
        forward_time = time()
        loss, loss_stats = self.compute_loss(model_batch, no_model_batch)
        stats.update({k:v for k,v in loss_stats.items() if "NO_LOGGING" not in k})
        torch.cuda.synchronize()
        forward_time = time() - forward_time

        # backward
        backward_time = time()
        self.backward(loss, loss_stats)
        torch.cuda.synchronize()
        backward_time = time() - backward_time

        self.grad_norm = 0.0
        if self.model.is_gradient_accumulation_boundary():
            if self.args.fp32:
                self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            else:
                self.grad_norm = self.optimizer.scaled_global_norm() / self.optimizer.cur_scale
        
        noise_batch_stats = {}
        
        # step
        step_time = time()
        self.model.step()
        torch.cuda.synchronize()
        step_time = time() - step_time

        stats["loss"] = self._all_reduce_loss(loss)
            
        elapsed_time = forward_time + backward_time + step_time
        stats["elasped_time"] = elapsed_time

        return stats, noise_batch_stats

    def train(self):
        if self.args.do_train:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=((not self.args.precompute_data_order) and (not self.args.no_shuffle)), drop_last=True, rank=self.dp_rank, num_replicas=self.dp_world_size)
            train_dataloader = DataLoader(
                self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate, drop_last=True)

        self.steps = 0
        self.global_steps = 1
        self.epoch = 0
        
        logging_stats = defaultdict(float)
        if self.args.do_train and self.dp_rank == 0:
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
        
        if self.args.do_train and not self.args.resume_training and not self.args.no_eval_when_start:
            self.evaluate()

        if self.args.torch_compile:
            torch.compiler.cudagraph_mark_step_begin()

        st_time = time()
        
        assert self.epochs is not None
        assert self.total_steps is not None
        
        for epoch in range(0, self.epochs):
            self.set_train()
            self.epoch = epoch
            train_sampler.set_epoch(epoch)
            self.train_dataset.set_epoch(epoch)
            self.print_and_save("New Epoch")
            for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
                if self.args.resume_training or self.args.start_from_global_step is not None:
                    if self.global_steps <= self.last_global_steps:
                        if (self.steps % self.args.gradient_accumulation_steps == 0) and (self.global_steps % 1000 == 0):
                            print_rank(f"Skipping global step {self.global_steps}")                        
                        self.steps += 1
                        if self.steps % self.args.gradient_accumulation_steps == 0:
                            self.global_steps += 1
                        continue
                    if (self.steps % self.args.gradient_accumulation_steps == 0):
                        print_rank(f"Starting from global step {self.global_steps}")
                        if self.args.resume_training:
                            torch.set_rng_state(self.last_rng_states["torch"])
                            torch.cuda.set_rng_state(self.last_rng_states["cuda"])
                            np.random.set_state(self.last_rng_states["numpy"])
                            random.setstate(self.last_rng_states["python"])

                if not self.first_printed:
                    self.first_print(model_batch, no_model_batch, "train")
                    self.first_printed = True

                self.train_dataset.move_to_device(model_batch, no_model_batch, self.device)
                                
                stats = {}
                
                stats, noise_batch_stats = self._train_pass(model_batch, no_model_batch, stats)

                # logging
                for k in stats:
                    logging_stats[k] += stats[k]
                
                mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                
                # print first step
                if self.steps == 0:
                    print_rank(self.get_log(stats, "train",
                        lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                        scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)
                    print_rank("-" * 100)
                    print_rank("-" * 100)
                
                if (self.args.mid_log_num > 0) and ((self.steps+1) % mid_log_step == 0):
                    print_rank(self.get_log(stats, "train",
                                            lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                            scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0),)

                
                if (self.args.gradient_accumulation_steps == 1 or self.steps > 0) and \
                    (self.global_steps > 0) and \
                        (self.global_steps % self.args.log_interval == 0) and \
                             ((self.steps+1) % self.args.gradient_accumulation_steps == 0):
                    logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                    now_time = time()
                    real_step_time = (now_time - st_time) / self.args.log_interval
                    st_time = now_time
                    log_str = self.get_log(logging_stats, "train", 
                                           grad_norm="{:.4f}".format(self.grad_norm),
                                           **noise_batch_stats,
                                           lr="{:.4e}".format(self.lr_scheduler.get_last_lr()[0]),
                                           scale=self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0,
                                           step_time=logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps,
                                           real_step_time = real_step_time)
                    
                    if self.dp_rank == 0:
                        wandb_logging_stats = {
                            **logging_stats,
                            "grad_norm": self.grad_norm,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "scale": self.optimizer.cur_scale if hasattr(self.optimizer, "cur_scale") else 0,
                            "step_time": logging_stats.get("elasped_time", 0) * self.args.gradient_accumulation_steps,
                        }
                        
                        wandb.log(wandb_logging_stats, step=self.global_steps)
                    
                    print_rank("*" * 100)
                    self.print_and_save(log_str)
                    print_rank(self.args.save)
                    print_rank("*" * 100)
                    logging_stats = {k:0 for k in logging_stats}
                    
                    # exit(0)

                # save
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    ((self.global_steps % self.args.save_interval == 0) or self.global_steps in [10, 100, 1000]):
                    self.save(self.args.save)

                # eval
                if (self.steps > 0) and (self.global_steps > 0) and ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and \
                    (self.global_steps % self.args.eval_interval == 0):
                    self.evaluate()
                    self.set_train()

                # end
                if ((self.steps+1) % self.args.gradient_accumulation_steps == 0) and (self.global_steps >= self.total_steps):
                    self.save(self.args.save)
                    self.evaluate()
                    return
                
                self.steps += 1
                if self.steps % self.args.gradient_accumulation_steps == 0:
                    self.global_steps += 1

        if self.args.do_infer:
            self.inference()
        
        if self.args.do_eval:
            self.evaluate()

        if self.args.do_train and self.dp_rank == 0:
            run.finish()

    def inference(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _avg_loss_cross_dp(self, all_losses):
        all_losses = all_gather(all_losses, dim=1, group=self.dp_group, world_size=self.dp_world_size, op="stack")
        all_losses = all_losses.view(-1)
        avg_loss = all_losses.mean().item()
        return avg_loss

    def evaluate_lm(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=eval_dataset.collate)
        
        self.model.eval()
        all_losses = []
                    
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"LM Evaluation", disable=(not self.dp_rank == 0))):
                if i == 0 and self.dp_rank == 0:
                    self.first_print(model_batch, no_model_batch, f"eval_{eval_dataset.data_name}")
                eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
                all_losses.append(loss)
                if i % 20 == 0:
                    print_rank(f"{i}/{len(eval_dataloader)}")
        
        all_losses = torch.cat(all_losses, dim=0)
        avg_loss = self._avg_loss_cross_dp(all_losses)

        if self.dp_rank == 0:
            res = {"avg_loss": avg_loss}
        else:
            res = None
        
        dist.barrier()
        return res
    
    def get_generattion_config(self, batch):
        max_new_tokens = self.args.max_length - batch["input_ids"].size(1)
        generation_config = GenerationConfig(
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
            max_length=self.args.max_length,
            min_length=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        return generation_config
    
    def generate(self, batch, decode_type="trm_ar"):
        generation_config = self.get_generattion_config(batch)
        gen_out = self.model.generate(**batch, generation_config=generation_config)
        return gen_out
        
    def save_evals(self, preds, results, response_texts, directory = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        save_dir = os.path.join(base_ckpt_path, "eval", f"{self.global_steps}")
        os.makedirs(save_dir, exist_ok=True)
        
        if self.dp_rank == 0:
            torch.save(preds, os.path.join(save_dir, "preds.pt"))
            torch.save(results, os.path.join(save_dir, "results.pt"))
            with open(os.path.join(save_dir, "answers.jsonl"), "w") as f:
                for resp in response_texts:
                    f.write(json.dumps({"text": resp}) + "\n")

    def save(self, directory):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{self.global_steps}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.args.model_parallel:
            if self.dp_rank == 0:
                self.model.module.config.to_json_file(os.path.join(ckpt_dir, "config.json"))
                self.tokenizer.save_pretrained(ckpt_dir)
            if mpu.get_data_parallel_rank() == 0:
                save_parallel(self.model.module.base_model, ckpt_dir)
        else:
            if self.dp_rank == 0:
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)

            if self.args.save_all:
                self.model.save_checkpoint(base_ckpt_path, tag=f"{self.global_steps}")
                rng_states = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }
                torch.save(rng_states, os.path.join(ckpt_dir, f"rng_states_{self.dp_rank}.pt"))
                if self.dp_rank == 0:
                    with open(os.path.join(ckpt_dir, "dynamics.json"), "w") as f:
                        json.dump({
                            "step": self.steps,
                            "epoch": self.epoch,
                            "global_steps": self.global_steps,
                            "skip_offset": (self.epoch, self.global_steps * self.total_batch_size)
                        }, f)
            else:
                if self.dp_rank == 0:
                    self.model.module.save_pretrained(ckpt_dir, safe_serialization=False)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
        
        dist.barrier()