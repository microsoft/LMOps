from typing import Callable

import torch
import random
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """
    Orchestrator that prepares data for PPO training: transforms samples from `pipeline` into `PPOBatch` and pushes them into model's `store`
    """

    def __init__(
        self,
        model: BaseRLModel,
        pipeline: BasePipeline,
        reward_fn: Callable,
        metric_fn: Callable = None,
        chunk_size: int = 512,
    ):
        self.pipeline = pipeline
        self.rl_model = model
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True
        )
        self.pipeline_loader = self.rl_model.accelerator.prepare(self.pipeline_loader)
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.rl_model.model, "frozen_head"):
            self.ref_model = self.rl_model.get_arch(self.rl_model.config)

        self.rl_model.orch = self
        self.rl_model.reward_fn = reward_fn
        self.rl_model.metric_fn = metric_fn
        self.min_new_tokens = 15
        self.max_new_tokens = 75

    def score(self, samples, plain_aes_score=None):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples, plain_aes_score=plain_aes_score)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            # print("batch=")
            # print(batch)

            # diverse beam search
            query_tensors = batch.input_ids
            plain_aes_score = batch.aes_score.cpu() if hasattr(batch, "aes_score") else None 
            
            tmp_max_new_tokens = random.randint(self.min_new_tokens, self.max_new_tokens)
            self.rl_model.generate_kwargs["max_new_tokens"] = tmp_max_new_tokens
            samples = self.rl_model.generate(**batch)
            num_return_sequences = self.rl_model.generate_kwargs["num_return_sequences"]
            sample_index = torch.randint(0, num_return_sequences, (query_tensors.shape[0],))
            samples = samples.reshape(query_tensors.shape[0], num_return_sequences, -1)
            samples = samples[torch.arange(query_tensors.shape[0]), sample_index]

            # add an extra padding
            samples = torch.cat([samples, torch.ones_like(samples[:,0:1])*self.rl_model.tokenizer.pad_token_id], -1)

            response_tensors = samples[:, query_tensors.shape[1] :]
            texts = self.rl_model.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            scores = torch.as_tensor(self.score(texts, plain_aes_score))

            # Precompute logprobs, values
            all_tokens = torch.cat(
                (query_tensors.to(samples.device), response_tensors), dim=1
            )

            attention_mask = (
                all_tokens.not_equal(self.rl_model.tokenizer.pad_token_id)
                .long()
                .to(all_tokens.device)
            )

            # for a proper positional encoding in case of left padding
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

            with torch.no_grad():
                # logits, _, v = self.rl_model.model(all_tokens)
                logits, _, v = self.rl_model.model(all_tokens, attention_mask, position_ids=position_ids)
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.rl_model.model, "frozen_head"):
                    # TODO 1 check whether hydra head works
                    # TODO 2. the efficiency of this model, maybe forward too many times (3 times)
                    ref_logits = self.rl_model.model.forward_hydra(
                        all_tokens, attention_mask=attention_mask, return_dict=False, position_ids=position_ids)
                # fix bug: 
                elif hasattr(self.rl_model.model, "module"):
                    ref_logits = self.rl_model.model.module.forward_hydra(
                        all_tokens, attention_mask=attention_mask, return_dict=False, position_ids=position_ids)
                else:
                    raise NotImplementedError

            ref_logits = ref_logits.to(self.rl_model.accelerator.device)
            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )
            start = query_tensors.size()[1] - 1
            end = query_tensors.size()[1] + response_tensors.size()[1] - 1
            all_values = v[:, start:end]
            all_logprobs = logprobs[:, start:end]
            all_ref_logprobs = ref_logprobs[:, start:end]

            response = all_tokens[:, 1:][:, start:end]
            padding_mask = response == self.rl_model.tokenizer.pad_token_id
            padding_num = padding_mask.long().sum(-1)

            if (padding_num == 0).any():
                padding_num[padding_num==0] = response.shape[-1] - 1
            padding_mask.scatter_(-1, (response.shape[-1]-padding_num).unsqueeze(-1), False)

            # Compute rewards
            kls = all_logprobs - all_ref_logprobs
            kls[padding_mask] = 0.0
            all_values[padding_mask] = 0.0

            non_score_rewards = -self.rl_model.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()

            all_rewards.scatter_(-1, (end-start-padding_num).unsqueeze(-1), scores.to(self.rl_model.accelerator.device).unsqueeze(-1))

            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            all_logprobs = all_logprobs.cpu()
            all_values = all_values.cpu()
            all_rewards = all_rewards.cpu()
            scores = scores.cpu()

            if "input_ids_mixin" in batch:
                input_ids_mixin = batch.input_ids_mixin.cpu()
                attention_mask_mixin = batch.attention_mask_mixin.cpu()
                token_type_ids_mixin = batch.token_type_ids_mixin.cpu()
            else:
                input_ids_mixin, attention_mask_mixin, token_type_ids_mixin = None, None, None

            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i, :],
                    response_tensor=response_tensors[i, :],
                    logprobs=all_logprobs[i, :],
                    values=all_values[i, :],
                    rewards=all_rewards[i, :],
                    score_train=scores[i],
                    input_ids_mixin=input_ids_mixin[i, :] if input_ids_mixin is not None else None,
                    attention_mask_mixin=attention_mask_mixin[i, :] if attention_mask_mixin is not None else None,
                    token_type_ids_mixin=token_type_ids_mixin[i, :] if token_type_ids_mixin is not None else None,
                )
                for i in range(query_tensors.size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        stats = {"exp_time": exp_time}
        self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
