import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchtyping import TensorType

from .data_types import PPORLBatch
from .utils import whiten, get_entropy, get_x_entropy, get_log_probs

from transformers import mpu

from utils import all_gather, print_rank


class Loss():
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer

    def _get_cumsum_rewards(self, rewards):          
        full_rewards = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(rewards.size(1))):
            full_rewards = self.args.gamma * full_rewards + rewards[:, t]
            
        return full_rewards

    def _get_advantages_and_returns(
        self,
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        mask: TensorType["batch_size", "response_size"],
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_rw = 0
        rw_reversed = []
        
        rewards = rewards.float()
        mask = mask.float()
        lens = torch.cumsum(mask, dim=-1)      # faster way        
        lens = mask - lens + lens[:, -1:None]  # faster way
        lens = torch.masked_fill(lens, lens==0, 1)

        for t in reversed(range(response_length)):
            rw_delta = rewards[:, t]
            last_rw = rw_delta + self.args.gamma * last_rw
            rw_reversed.append(last_rw)

        rw = torch.stack(rw_reversed[::-1], dim=1)
        rw = rw / lens

        advantages = rw

        if use_whitening:
            advantages = whiten(advantages)
        
        return advantages.detach()

    def _pg_loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
        w: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        n = mask.sum()
        
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio.float())            
        ratio = ratio * w

        if any(torch.isinf(advantages).view(-1)):
            print("[ERROR] advantage inf")
        
        if any(torch.isinf(ratio).view(-1)):
            print("[ERROR] ratio inf")

        if any(torch.isnan(advantages).view(-1)):
            print("[ERROR] advantage nan")
        
        if any(torch.isnan(ratio).view(-1)):
            print("[ERROR] ratio nan")
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.args.cliprange,
            1.0 + self.args.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2).float() * mask) / n

        return pg_loss

    def _reg_loss(self, query_ids, response_ids, mask, logits, inf_mask, stats):
        with torch.no_grad():
            t_logits = self.trainer.compute_logits_and_log_probs(query_ids, response_ids, inf_mask, base="teacher", return_logprobs=False)
        
        loss_exp_ent = 0
        xent = get_x_entropy(logits, t_logits, inf_mask, mask, model_parallel=self.args.model_parallel)
        s_ent = get_entropy(logits, inf_mask, mask, model_parallel=self.args.model_parallel)
        loss_exp_ent = torch.sum((xent - s_ent) * mask) / mask.sum()
        stats["reg_loss"] = loss_exp_ent.item()
        
        return loss_exp_ent

    def get_input_batch(self, ppo_batch: PPORLBatch, pt_batch):
        query_tensors = ppo_batch.query_tensors
        response_tensors = ppo_batch.response_tensors
        ppo_input_batch = self.trainer.get_model_inputs(query_tensors, response_tensors)
        pt_input_batch, _ = pt_batch
        # merge batch
        assert len(ppo_input_batch) == len(pt_input_batch), list(ppo_input_batch.keys())
        input_batch = {}
        for k in ppo_input_batch:
            input_batch[k] = torch.cat([ppo_input_batch[k], pt_input_batch[k]], dim=0)
        return input_batch

    def ppo_loss(self, batch: PPORLBatch, logits):
        stats = {}
        query_tensors = batch.query_tensors
        response_tensors = batch.response_tensors
        lens = batch.lens
        s_lens = batch.s_lens
        mask = batch.mask
        old_logprobs = batch.logprobs
        old_rewards = batch.rewards
        rev_kl = batch.rev_kl
        w = batch.w
        inf_mask = batch.inf_mask
        
        response_length = response_tensors.shape[-1]

        start = query_tensors.size(1) - 1 # "-1" for the first generated token AS TARGET
        end = query_tensors.size(1) + response_tensors.size(1) - 1 # "remove the last token that does not have target"

        logits = logits / self.args.temperature
        logits = logits[:, start:end]
        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))
            
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.trainer.max_length :
        ]
        mask = self.trainer.get_mask(tokens)[:, start:end]
        
        logprobs = get_log_probs(logits, response_tensors, mask, inf_mask, model_parallel=self.args.model_parallel)

        advantages = self._get_advantages_and_returns(
            old_rewards, response_length, mask
        )
        
        loss = self._pg_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            w=w,
        )
        stats["pg_loss"] = loss.item()
        
        single_step_reg_loss = self._reg_loss(query_tensors, response_tensors, mask, logits, inf_mask, stats)
        stats["reg_loss"] = single_step_reg_loss.item()
        
        if self.args.single_step_reg:
            loss += single_step_reg_loss
        
        stats["rl_loss"] = loss.item()
        
        with torch.no_grad():
            # generation values for reward
            cumsum_rewards = self._get_cumsum_rewards(old_rewards)
            rev_kl = torch.sum(rev_kl, dim=-1)
            
            if self.args.length_norm:
                cumsum_rewards = cumsum_rewards / lens
                rev_kl = rev_kl / s_lens
                        
            cumsum_rewards = all_gather(cumsum_rewards, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).mean(dim=0).item()
            rev_kl = all_gather(rev_kl, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).mean(dim=0).item()
            lens = all_gather(lens, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).float().mean(dim=0).item()
            s_lens = all_gather(s_lens, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).float().mean(dim=0).item()
        
        stats["reward"] = cumsum_rewards
        stats["rev_kl"] = rev_kl
        stats["mixed_lens"] = lens
        stats["stu_lens"] = s_lens
        
        return loss, stats

    def pt_loss(self, batch, logits):
        stats = {}
        model_batch, no_model_batch = batch
        loss_mask = (no_model_batch["label"] != -100).int()
        if self.args.model_parallel:
            lm_losses = mpu.parallel_cross_entropy(logits.contiguous().float(), no_model_batch["label"]).view(-1)
            lm_loss = (lm_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
        
        distil_loss = 0
        if self.trainer.teacher_model is not None and self.args.kd_ratio is not None:
            with torch.no_grad():
                teacher_outputs = self.trainer.teacher_model(**model_batch, return_dict=True, use_cache=False)
                teacher_logits = teacher_outputs.logits
            if self.args.model_parallel:
                distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
                distil_losses = distil_losses.view(-1)
                distil_loss = (distil_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
            else:
                teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
                inf_mask = torch.isinf(logits)
                logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
                x = torch.sum(prod_probs, dim=-1).view(-1)
                distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
            
            loss = (1-self.args.kd_ratio) * lm_loss + self.args.kd_ratio * distil_loss

        stats["pt_loss"] = loss.item()
        stats["lm_loss"] = lm_loss.item()
        stats["ds_loss"] = distil_loss.item()

        return loss, stats