import torch
import os

from .data_types import PromptBatch, PPORLElement
from .pipelines import PPOPipeline
from .trainer import PPOTrainer

from utils import get_rank, print_rank, all_gather, save_rank
from .utils import get_rev_kl
from transformers import mpu

class PPOSampler():
    """
    Orchestrator prepares data for PPO training.
    Transforms samples from `pipeline` into `PPOBatch` and pushes them into trainer's `store`
    """

    def __init__(
        self,
        args,
        trainer: PPOTrainer,
        pipeline: PPOPipeline,
        chunk_size: int = 512,
    ):
        self.args = args
        self.pipeline = pipeline
        self.trainer = trainer
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True, drop_last=True, num_workers=self.args.num_workers
        )
        self.pipeline_iterator = iter(self.pipeline_loader)

        self.trainer.set_sampler(self)

        self.epochs = 0

    def run_sample(self, num_rollouts_per_device: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts_per_device` prompts from `pipeline`, samples model and computes the
        KL againts a reference model. It then appends PPOElements to trainer's `store`
        """
        ppo_rl_elements = []

        while len(ppo_rl_elements) < num_rollouts_per_device:
            if ((not self.args.model_parallel) or mpu.get_model_parallel_rank()) == 0:
                print(f"Rank {get_rank()}: Number Sampling Elements {len(ppo_rl_elements)} / {num_rollouts_per_device}")
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.epochs += 1
                print_rank(f"Another outer ppo epoch, outer ppo epoch: {self.epochs}")
                save_rank(f"Another outer ppo epoch, outer ppo epoch: {self.epochs}", os.path.join(self.args.save, "log.txt"))
                
                self.pipeline_loader.sampler.set_epoch(self.epochs)
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            batch, no_model_batch = batch
            n = batch["input_ids"].size(0)
            
            batch, no_model_batch = self.pipeline.move_to_device(batch, no_model_batch, self.trainer.device)
            
            query_ids = batch["input_ids"]
            
            # generate and compute rollout scores
            with torch.no_grad():
                mode = "base"
                gen_out = self.trainer.generate(**batch, return_dict_in_generate=True, mode=mode, teacher_mixed_sample=(self.args.teacher_mixed_alpha is not None), output_scores=True)
                full_ids = gen_out.sequences
                response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
                mask = (full_ids != self.trainer.tokenizer.pad_token_id)[:, query_ids.size(-1)-1:query_ids.size(-1)+response_ids.size(-1)-1]
                lens = torch.sum(mask, dim=-1)
                gen_logits = gen_out.scores # NOTE: [b, s, h_p]
                inf_mask = torch.isinf(gen_logits)
                scores = self.trainer.reward_fn(query_ids, response_ids, inf_mask=inf_mask)
                t_rewards = scores["rewards"]
                inf_mask = scores["inf_mask"]
                _, rollout_logprobs = self.trainer.compute_logits_and_log_probs(query_ids, response_ids, inf_mask=inf_mask, base=mode)

                # student generation features
                if self.args.teacher_mixed_alpha is not None:
                    s_gen_out = self.trainer.generate(**batch, return_dict_in_generate=True, mode=mode, output_scores=True)
                    s_full_ids = s_gen_out.sequences
                    s_response_ids = s_full_ids[:, query_ids.size(1):]
                    s_inf_mask = torch.isinf(s_gen_out.scores)
                    s_response_ids = s_full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
                    s_scores = self.trainer.reward_fn(query_ids, s_response_ids, inf_mask=s_inf_mask)
                    s_t_rewards = s_scores["rewards"]
                    s_inf_mask = s_scores["inf_mask"]
                    _, s_rollout_logprobs = self.trainer.compute_logits_and_log_probs(query_ids, s_response_ids, inf_mask=s_inf_mask, base=mode)
                    s_mask = (s_full_ids != self.trainer.tokenizer.pad_token_id)[:, query_ids.size(-1)-1:query_ids.size(-1)+s_response_ids.size(-1)-1]
                    s_lens = torch.sum(s_mask, dim=-1)
                else:
                    s_t_rewards = t_rewards
                    s_rollout_logprobs = rollout_logprobs
                    s_mask = mask
                    s_lens = lens

            rev_kl = get_rev_kl(s_t_rewards, s_rollout_logprobs, s_mask)

            if self.args.teacher_mixed_alpha is not None:
                with torch.no_grad():
                    _, t_rollout_logprobs = self.trainer.compute_logits_and_log_probs(query_ids, response_ids, inf_mask=inf_mask, base="teacher") # recompute because of the fp16 loss

            # get logprobs and the importance sampling weight w
            with torch.no_grad():
                if self.args.teacher_mixed_alpha is not None:
                    _, raw_logprobs = self.trainer.compute_logits_and_log_probs(query_ids, response_ids, inf_mask=inf_mask, base="base") # raw_logprobs: compute using the new model
                    logprobs = raw_logprobs
                    mix_probs = (1 - self.args.teacher_mixed_alpha) * torch.exp(rollout_logprobs.float()) + self.args.teacher_mixed_alpha * torch.exp(t_rollout_logprobs.float())
                    mix_logprobs = torch.log(mix_probs)
                    log_w = logprobs - mix_logprobs
                    w = torch.exp(log_w) # importance sampling weight
                else:
                    raw_logprobs = rollout_logprobs
                    logprobs = rollout_logprobs
                    w = torch.ones_like(logprobs)
                        
                # get ent_rewards
                ent_rewards = -logprobs

            rewards = t_rewards + ent_rewards

            if self.args.reward_scaling is not None:
                rewards = rewards / self.args.reward_scaling

            clip_reward = self.args.cliprange_reward
            if clip_reward:
                rewards = torch.clip(rewards, -clip_reward, clip_reward)

            query_ids = query_ids.cpu()
            response_ids = response_ids.cpu()
            lens = lens.cpu()
            s_lens = s_lens.cpu()
            mask = mask.cpu()
            logprobs = logprobs.cpu()
            rewards = rewards.cpu()
            rev_kl = rev_kl.cpu()
            w = w.cpu()
            inf_mask = inf_mask.cpu()
            
            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_ids[i],
                    response_tensor=response_ids[i],
                    lens=lens[i],
                    s_lens=s_lens[i],
                    mask=mask[i],
                    logprobs=logprobs[i],
                    rewards=rewards[i],
                    rev_kl=rev_kl[i],
                    w=w[i],
                    inf_mask=inf_mask[i],
                    t_rewards=t_rewards[i],
                    ent_rewards=ent_rewards[i]
                )
                for i in range(n)
            ]
            ppo_rl_elements.extend(new_ppo_rl_elements)

        ppo_rl_elements = ppo_rl_elements[:num_rollouts_per_device]
        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
        
        if self.args.save_rollout:
            all_query_ids = all_gather(torch.stack([e.query_tensor for e in ppo_rl_elements], dim=0).to(self.trainer.device))
            all_response_ids = all_gather(torch.stack([e.response_tensor for e in ppo_rl_elements], dim=0).to(self.trainer.device))
            all_entropy = all_gather(torch.stack([e.entropy for e in ppo_rl_elements], dim=0).to(self.trainer.device))
            rollout_save_path = os.path.join(self.args.save, "rollout_history", str(iter_count))
            if get_rank() == 0:
                os.makedirs(rollout_save_path, exist_ok=True)
                torch.save((all_query_ids, all_response_ids, all_entropy), os.path.join(rollout_save_path, "all.pt"))
