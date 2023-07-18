import importlib
import os
from abc import abstractmethod
from time import time
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer

import wandb
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model

if importlib.util.find_spec("rich") is not None:
    from tqdm.rich import tqdm
else:
    from tqdm import tqdm


@register_model
class AccelerateRLModel(BaseRLModel):
    """
    RL Model that uses accelerate for training
    """

    def __init__(self, config, train_mode=True):
        super().__init__(config, train_mode)

        self.accelerator = Accelerator(log_with="wandb", logging_dir=config.train.logging_dir)

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        else:
            torch.random.manual_seed(config.train.seed)

        # Retrieves model equipped for ppo, ilql, etc
        self.model = self.get_arch(self.config)
        self.max_length = config.train.seq_length
        self.eval_only = config.train.eval_only
        self.iter_count = 0

        if config.model.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer = None

        if hasattr(self.model.gpt, "gpt_neox"):
            gpt_blocks = self.model.gpt.gpt_neox.layers
        else:
            gpt_blocks = self.model.gpt.transformer.h

        # freeze transformer's bottom layers if num_layers_unfrozen >= 0
        num_layers_unfrozen = self.config.model.num_layers_unfrozen
        if num_layers_unfrozen == 0:
            gpt_blocks_to_freeze = list(gpt_blocks)
        elif num_layers_unfrozen > 0:
            gpt_blocks_to_freeze = list(gpt_blocks)[:-num_layers_unfrozen]
        else:
            gpt_blocks_to_freeze = []

        for m in gpt_blocks_to_freeze:
            m.requires_grad_(False)

        if self.accelerator.is_main_process and not self.eval_only:
            self.accelerator.init_trackers(
                project_name=self.config.train.project_name,
                config=self.config.to_dict(),
                init_kwargs={
                    "wandb": {
                        "name": f"{config.model.model_path}",
                        "dir": config.train.logging_dir,
                        "mode": "disabled"
                        if os.environ.get("debug", False)
                        else "online",
                    }
                },
            )

        if not self.eval_only:
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.train.learning_rate_init),
                betas=self.config.train.opt_betas,
            )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                self.config.train.total_steps,
                eta_min=float(self.config.train.learning_rate_target),
            )

    def tokenize(self, text: Iterable[str]):
        """
        Tokenize a batch of text after adding bos token to each of the samples
        """
        text = [self.tokenizer.bos_token + txt for txt in text]
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.seq_length,
            return_tensors="pt",
        )

    def generate(self, input_ids, attention_mask=None, aes_score=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def get_components(self) -> Dict[str, any]:
        components = (
            {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
            if self.train_mode
            else {"model": self.model}
        )
        return components

    def save(self, directory=None):
        """Creates checkpoint of optimizer, scheduler and a model"""
        if self.accelerator.is_main_process:
            directory = directory or self.config.train.checkpoint_dir
            rank = self.accelerator.process_index
            directory = os.path.join(directory, "r%d-step%d" % (rank, self.iter_count))
            print("[I] Save states to %s" % directory)
            os.makedirs(directory, exist_ok=True)
            try:
                self.accelerator.save_state(directory)
            except OSError:
                print("[W] OSError while saving states...")

    def resume_training(self):
        directory = os.path.join(self.config.train.checkpoint_dir, "ckpt")
        if not os.path.exists(directory) or len(os.listdir(directory)) == 0:
            return 

        steps_all = os.listdir(directory)
        steps_all = [int(ckpt.replace("r0-step", "")) for ckpt in steps_all]
        self.iter_count = max(steps_all)
        max_step_ckpt = os.path.join(directory, "r0-step"+str(max(steps_all)))
        print(f"LOADING previous ckpt: {max_step_ckpt}")

        self.model.module.load_state_dict(torch.load(os.path.join(max_step_ckpt, "pytorch_model.bin")))
        self.opt.load_state_dict(torch.load(os.path.join(max_step_ckpt, "optimizer.bin")))
        self.scheduler.load_state_dict(torch.load(os.path.join(max_step_ckpt, "scheduler.bin")))

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""

        stats = {}
        all_samples = []
        all_aes_score = []
        generate_time = time()
        for prompts in self.eval_dataloader:
            if hasattr(prompts, "aes_score"):
                all_aes_score.append(prompts.aes_score.cpu())

            if isinstance(prompts, torch.Tensor):
                samples = self.generate(prompts)
            else:
                samples = self.generate(**prompts)

            num_return_sequences = self.generate_kwargs["num_return_sequences"]
            samples = samples.reshape(-1, num_return_sequences, samples.shape[-1])
            samples = samples[:,0,:]

            if isinstance(samples, tuple):
                samples, *_ = samples

            pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
            all_samples.append(
                F.pad(
                    samples,
                    (0, self.max_length - samples.shape[1]),
                    value=pad_token,
                )
            )
        stats["generate_time"] = time() - generate_time

        # samples = self.accelerator.gather(torch.vstack(all_samples))
        # NOTE walkaround for parallel eval
        samples = torch.vstack(all_samples)
        plain_aes_score = torch.cat(all_aes_score) if len(all_aes_score) > 0 else None
        if self.tokenizer:
            samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        if isinstance(samples[0], str):
            columns_data = [samples]
        else:
            columns_data = [samples.tolist()]
        columns = ["samples"]

        if self.reward_fn:
            rewards = torch.as_tensor(self.reward_fn(samples, plain_aes_score), dtype=torch.float, device=self.accelerator.device)
            all_rewards = self.accelerator.gather(rewards)
            if self.accelerator.is_main_process:
                print("[I] all_rewards.shape=")
                print(all_rewards.shape)
                mean_reward = all_rewards.mean()
                columns.append("reward")
                columns_data.append(all_rewards)
                stats["mean_reward"] = mean_reward
                print(f"mean_reward={mean_reward}")

        if self.metric_fn:
            raise NotImplementedError
        
        if self.accelerator.is_main_process and not self.eval_only:
            rows = list(zip(*columns_data))
            stats["samples"] = wandb.Table(columns=columns, rows=rows)

            print(rows[0])

        return stats
    
    def evaluate_deprecated(self):
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""

        stats = {}
        all_samples = []
        generate_time = time()
        for prompts in self.eval_dataloader:
            if isinstance(prompts, torch.Tensor):
                samples = self.generate(prompts)
            else:
                samples = self.generate(**prompts)

            num_return_sequences = self.generate_kwargs["num_return_sequences"]
            samples = samples.reshape(-1, num_return_sequences, samples.shape[-1])
            samples = samples[:,0,:]

            if isinstance(samples, tuple):
                samples, *_ = samples

            pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
            all_samples.append(
                F.pad(
                    samples,
                    (0, self.max_length - samples.shape[1]),
                    value=pad_token,
                )
            )
        stats["generate_time"] = time() - generate_time

        # samples = self.accelerator.gather(torch.vstack(all_samples))
        # NOTE walkaround for parallel eval
        samples = torch.vstack(all_samples)

        if self.accelerator.is_main_process:
            if self.tokenizer:
                samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)

            if isinstance(samples[0], str):
                columns_data = [samples]
            else:
                columns_data = [samples.tolist()]
            columns = ["samples"]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                rewards = torch.as_tensor(self.reward_fn(samples), dtype=torch.float)
                mean_reward = rewards.mean()
                columns.append("reward")
                columns_data.append(rewards)
                stats["mean_reward"] = mean_reward
                print(f"{mean_reward}")

            # additionally log any other metrics
            if self.metric_fn:
                metric_time = time()
                metrics = self.metric_fn(samples)
                stats["metric_time"] = time() - metric_time

                mean_metrics = {
                    f"metrics/{k}": torch.as_tensor(xs).mean(-1)
                    for k, xs in metrics.items()
                }

                stats.update(mean_metrics)

                for metric, values in metrics.items():
                    columns.append(metric)
                    columns_data.append(values)

            rows = list(zip(*columns_data))
            stats["samples"] = wandb.Table(columns=columns, rows=rows)

            print(rows[0])

        return stats

    def learn(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.prepare_learning()

        tbar = tqdm(
            total=self.total_steps, disable=not self.accelerator.is_local_main_process
        )

        for _ in range(self.config.train.epochs):
            for _ in range(self.n_updates_per_batch):
                for batch in self.train_dataloader:
                    forward_time = time()
                    loss, stats = self.loss(batch)
                    forward_time = time() - forward_time

                    backward_time = time()
                    self.accelerator.backward(loss)
                    if self.config.train.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)
                    backward_time = time() - backward_time

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if self.iter_count % self.config.train.checkpoint_interval == 0:
                        self.save()

                    if self.iter_count % self.config.train.eval_interval == 0:
                        results = self.evaluate()

                        results.update(stats)
                        results.update(
                            {
                                "forward_time": forward_time,
                                "backward_time": backward_time,
                            }
                        )
                        self.accelerator.log(results)

                    desc = ", ".join(f"{k}: {v:.2f}" for k, v in stats.items())
                    tbar.set_description(desc)
                    tbar.update()

                    if self.iter_count >= self.total_steps:
                        self.save()
                        return self.evaluate()

                self.post_backward_callback()

            self.post_epoch_callback()

    @abstractmethod
    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper of the decoder architecture"""
        pass

    @abstractmethod
    def loss(self, batch) -> Tuple[float, Dict]:
        """Compute loss on a batch from `store` and return some statistics"""
        pass

    @abstractmethod
    def post_backward_callback(self):
        """Do something after model update"""
        pass

    @abstractmethod
    def post_epoch_callback(self):
        """Do something after exhausting/single pass over `self.store`"""
        pass
