# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import json
import os
import pickle
import random
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.reward import get_medmcqa_acc, get_safety_acc
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask
import time

from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict
import verl.utils.torch_functional as verl_F

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    ExpLearner = 7


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def merge_topk_indices(indices1, indices2, target_k, special_marker=-1):
    """
    Merge two sets of topk indices into a combined set of size target_k.
    
    This function merges vocabulary indices from two different sources (e.g., actor and reference policy),
    removes duplicates, and pads the result to a fixed size with special markers.
    
    Args:
        indices1: torch.Tensor of shape (bs, seq_len, k1) - first set of indices
        indices2: torch.Tensor of shape (bs, seq_len, k2) - second set of indices
        target_k: int - target size (typically 2 * original k)
        special_marker: int - value to use for padding positions (default: -1)
        
    Returns:
        merged_indices: torch.Tensor of shape (bs, seq_len, target_k) - merged indices with special markers
        
    Example:
        >>> indices1 = torch.tensor([[[1, 3, 5]]])  # (1, 1, 3)
        >>> indices2 = torch.tensor([[[3, 7, 9]]])  # (1, 1, 3)
        >>> merged = merge_topk_indices(indices1, indices2, target_k=6, special_marker=-1)
        >>> merged  # (1, 1, 6) containing [1, 3, 5, 7, 9, -1]
    """
    assert indices1.shape[:-1] == indices2.shape[:-1], \
        f"Batch and sequence dimensions must match: {indices1.shape} vs {indices2.shape}"
    
    bs, seq_len = indices1.shape[0], indices1.shape[1]
    device = indices1.device
    dtype = indices1.dtype
    
    # Concatenate along the last dimension
    combined = torch.cat([indices1, indices2], dim=-1)  # (bs, seq_len, k1 + k2)
    
    # Initialize output tensor filled with special markers
    merged_indices = torch.full((bs, seq_len, target_k), special_marker, dtype=dtype, device=device)
    
    # Process each sequence position to remove duplicates
    for i in range(bs):
        for j in range(seq_len):
            # Get unique indices for this position
            unique_indices = torch.unique(combined[i, j])
            n_unique = unique_indices.shape[0]
            
            # Fill in as many unique indices as possible (up to target_k)
            n_fill = min(n_unique, target_k)
            merged_indices[i, j, :n_fill] = unique_indices[:n_fill]
    
    return merged_indices


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # Check if we're in textgame mode
        self.setting = config.trainer.get("setting", "default")
        self.is_textgame = self.setting == "textgame"
        # Initialize textgame environments if needed
        if self.is_textgame:
            self._init_textgame_envs()
        
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()

        # In textgame mode, we do not rely on datasets / dataloaders.
        if self.is_textgame:
            self._init_textgame_training()
        else:
            self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
    
    def _init_textgame_envs(self):
        """Initialize textgame environment configuration (not actual envs to avoid serialization)"""
        # Store environment configuration instead of actual environment objects
        # This avoids Ray serialization issues
        self.textgame_env_config = {
            'env_id': self.config.trainer.textgame_env_id,
            'num_players': 1,
            # Whether to keep the full response including reasoning before </think>
            'keep_reasoning': self.config.trainer.textgame_keep_reasoning,
            # Maximum prompt length fed into the textgame rollout tokenizer
            'max_prompt_length': self.config.trainer.textgame_max_prompt_length,
            # Maximum number of generated tokens for textgame actions
            'max_response_length': self.config.trainer.textgame_max_response_length,
            'no_think': self.config.trainer.textgame_no_think,
            'size': 3, # 4, 3
            'num_holes': 2,
            'randomize_start_goal': False,
            'num_boxes': 1,
            'dim_room': (6, 6),
        }

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _init_textgame_training(self):
        """
        Initialize training configuration for textgame mode.

        In this mode:
        - We do NOT need train/val datasets or dataloaders.
        - The total number of training steps is provided explicitly via `trainer.textgame_total_steps` instead of being derived from dataset size.
        """
        self.train_dataloader = None
        self.val_dataloader = None

        textgame_total_steps = self.config.trainer.get("textgame_total_steps", None)
        if textgame_total_steps is None:
            raise ValueError(
                "In textgame setting, `trainer.textgame_total_steps` must be set "
                "to control the total number of training steps."
            )

        self.total_training_steps = int(textgame_total_steps)
        print(f"[textgame] Total training steps set to {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps for textgame in config. Error: {e}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, sample_inputs=None, sample_outputs=None, teacher_outputs=None, gpt_outputs=None, dump_path=None, **kwargs):
        """Dump rollout/validation samples as JSONL."""
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

        if sample_inputs is not None:
            # LMSYS style
            val_files = self.config.data.val_files
            if isinstance(val_files, (list, tuple)) or OmegaConf.is_list(val_files):
                val_files = val_files[0]
            # Assuming path format: /path/to/{name}_gpt5_chat_test.parquet
            basename = os.path.basename(val_files)
            prefix = basename.split("_gpt5_chat")[0]
            filename = os.path.join(dump_path, f"{prefix}_generation_results.jsonl")

            n = len(sample_inputs)
            base_data = {
                "input": sample_inputs,
                "output": sample_outputs,
                "teacher_output": teacher_outputs,
                "gpt_output": gpt_outputs
            }
        else:
            # Old style (rollout dumping)
            inputs = kwargs.get("inputs")
            outputs = kwargs.get("outputs")
            scores = kwargs.get("scores")
            reward_extra_infos_dict = kwargs.get("reward_extra_infos_dict", {})
            
            filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
            n = len(inputs)
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.global_steps] * n,
            }
            if reward_extra_infos_dict:
                for k, v in reward_extra_infos_dict.items():
                    if len(v) == n:
                        base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _truncate_experience(self, experience, max_tokens):
        """Truncate experience from the end, keeping complete items based on markers."""
        exp_tokens = self.tokenizer.encode(experience, add_special_tokens=False)
        if len(exp_tokens) <= max_tokens:
            return experience

        exp_lines = experience.split("\n")
        truncated_lines = []
        current_tokens = 0

        def is_marker(line):
            if self.config.trainer.prompt_version == 'v2':
                return line.strip().startswith("- ")
            elif self.config.trainer.prompt_version in ['v1', 'v4']:
                return line.strip().startswith("- EXPERIENCE ITEM:")
            elif self.config.trainer.prompt_version in ['v3', 'v5']:
                return True
            return False

        for line in reversed(exp_lines):
            line_tokens = len(self.tokenizer.encode(line + "\n", add_special_tokens=False))
            if current_tokens + line_tokens > max_tokens:
                if is_marker(line) or not truncated_lines:
                    break
                continue
            else:
                truncated_lines.append(line)
                current_tokens += line_tokens
        
        truncated_lines = list(reversed(truncated_lines))
        start_idx = 0
        for i, line in enumerate(truncated_lines):
            if is_marker(line):
                start_idx = i
                break
        truncated_lines = truncated_lines[start_idx:]
        new_experience = "\n".join(truncated_lines)
        print(f"[Experience Truncate] Truncated from end, kept {len(self.tokenizer.encode(new_experience, add_special_tokens=False))} tokens")
        return new_experience

    
    def _build_dataproto(self, prompt_token_ids_list, response_ids_list):
        new_prompt_token_ids_list = []
        new_response_ids_list = []
        new_attention_mask = []
        new_position_ids = []
        for input_ids, response_ids in zip(prompt_token_ids_list, response_ids_list):    
            input_ids_ = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids_)
            input_ids_, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids_,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_ = input_ids_[0]
            attention_mask = attention_mask[0]
            position_ids = position_ids[0]
            
            new_prompt_token_ids_list.append(input_ids_)
            new_attention_mask.append(attention_mask)
            new_position_ids.append(position_ids)
            new_response_ids_list.append(response_ids)
        
        idx = torch.stack(new_prompt_token_ids_list)
        attention_mask = torch.stack(new_attention_mask)
        position_ids = torch.stack(new_position_ids)
        
        response = pad_2d_list_to_length(new_response_ids_list, self.tokenizer.pad_token_id, max_length=self.config.data.max_response_length)
        
        input_ids = torch.cat([idx, response], dim=-1)
        
        batch_size = idx.size(0)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        eos_token_id = self.tokenizer.eos_token_id
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        batch_dict = {
            "prompts": idx,
            "responses": response,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        batch_td = TensorDict(batch_dict, batch_size=batch_size)
        non_tensor_batch = {}
        
        return DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch)

    def _build_dataproto_woresp(self, prompt_token_ids_list):
        new_prompt_token_ids_list = []
        new_attention_mask = []
        new_position_ids = []
        for input_ids_s in prompt_token_ids_list:    
            input_ids_ = torch.tensor(input_ids_s).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids_)
            input_ids_, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids_,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_ = input_ids_[0]
            attention_mask = attention_mask[0]
            position_ids = position_ids[0]
            
            new_prompt_token_ids_list.append(input_ids_)
            new_attention_mask.append(attention_mask)
            new_position_ids.append(position_ids)
        
        input_ids = torch.stack(new_prompt_token_ids_list)
        attention_mask = torch.stack(new_attention_mask)
        position_ids = torch.stack(new_position_ids)
        batch_size = input_ids.size(0)
        
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        batch_td = TensorDict(batch_dict, batch_size=batch_size)
        non_tensor_batch = {}
        
        return DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch)

    def _re_tokenize(self, msgs):
        raw_prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=not self.textgame_env_config["no_think"])

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data["truncation"],
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        
        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0]
        }

    
    
    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        assert self.config.trainer.setting == "textgame"
        return self._validate_textgame()

    def _validate_textgame(self):
        
        """
        TextGame-only validation
        """

        assert getattr(self, "is_textgame", False), "_validate_textgame() called but self.is_textgame is False"
        eval_override = self.config.trainer.get("eval_override", False)

        # Check if we should skip experience generation and usage
        eval_wo_experience = self.config.trainer.get("eval_wo_experience", False)
        EXPERIENCE = ""
        EXPERIENCE_UPDATE_PROMPT = self.experience_update_prompt

        # NOTE: no EXPERIENCE_SOLVE_PROMPT_TEMPLATE here (unlike dataset-mode).

        sample_scores = []
        sample_response_lengths = []
        sample_avg_steps = []
        sample_avg_traj_tokens = []
        val_heldout_samples = []
        HELD_OUT_SIZE = int(self.config.trainer.get("held_out_size", 500))
        HELD_OUT_ROLLOUT = int(self.config.trainer.get("held_out_rollout", 2))
        val_samples_limit = int(self.config.trainer.get("val_samples_limit", 100))
        num_steps = int(self.config.trainer.textgame_max_steps)

        HELD_OUT_SIZE = HELD_OUT_SIZE - HELD_OUT_SIZE % self.actor_rollout_wg.world_size
        assert HELD_OUT_SIZE > 0
        # assert held_out_size is world_size's multiple
        assert HELD_OUT_SIZE % self.actor_rollout_wg.world_size == 0, f"held_out_size {HELD_OUT_SIZE} is not a multiple of world_size {self.actor_rollout_wg.world_size}"

        # NOTE: keep heldout unchanged
        heldout_seeds = [468382021 + 78025 + i * 1000 for i in range(HELD_OUT_SIZE)]
        training_seeds = [468382021 + 78025 + self.config.trainer.oel_round * 500000 + i * 1000 for i in range(HELD_OUT_SIZE, HELD_OUT_SIZE + val_samples_limit)]
        all_val_seeds = heldout_seeds + training_seeds

        print(f"[Validation TextGame] Generated {len(all_val_seeds)} seeds: {HELD_OUT_SIZE} held-out + {val_samples_limit} training")

        # If eval_wo_experience is True, only evaluate on HELD_OUT without experience
        if eval_wo_experience:
            # NOTE: we dont change heldout set across exp accu and test
            eval_prepend_experience = self.config.trainer.get("eval_prepend_experience", False)
            EXPERIENCE = ""
            if eval_prepend_experience:
                exp_path = self.config.trainer.experience_path
                with open(exp_path, "r", encoding="utf-8") as f:
                    EXPERIENCE = f.read().strip()
                max_exp_tokens = self.config.trainer.experience_max_length
                EXPERIENCE = self._truncate_experience(EXPERIENCE, max_exp_tokens)
                print(f"[Validation TextGame] eval_prepend_experience=True, loaded experience from {exp_path}")

            print(f"[Validation TextGame] eval_wo_experience=True, eval_prepend_experience={eval_prepend_experience}, evaluating on HELD_OUT only...")
            heldout_rewards_all = []

            for rr in range(HELD_OUT_ROLLOUT):
                # Seed offset to avoid identical stochasticity across rollouts
                rr_seeds = [s + 1000003 * rr for s in heldout_seeds]
                heldout_gen_output = self.actor_rollout_wg.generate_sequences_textgame(
                    env_config=self.textgame_env_config,
                    env_num=HELD_OUT_SIZE,
                    tokenizer=self.tokenizer,
                    experiences=[EXPERIENCE] * HELD_OUT_SIZE,
                    num_steps=num_steps,
                    seeds=rr_seeds,
                    validate=True
                )
                if isinstance(heldout_gen_output, list):
                    heldout_gen_output = heldout_gen_output[0]

                heldout_rewards = [reward_dict[0] for reward_dict in heldout_gen_output["reward_list"]]
                heldout_rewards_all.extend(heldout_rewards)

            heldout_reward_tensor = torch.tensor(heldout_rewards_all, dtype=torch.float32)
            acc = (heldout_reward_tensor == 1.0).float().mean().item()
            sample_scores.append(acc)
            
            # Calculate metrics
            all_response_token_counts = []
            all_traj_steps = []
            all_traj_tokens = []
            env_trajectories = heldout_gen_output.get("env_trajectories", {})
            for traj in env_trajectories.values():
                history = traj.get("history", [])
                all_traj_steps.append(len(history))
                traj_tokens = 0
                for step_info in history:
                    resp = step_info.get("raw_response", "")
                    tokens = self.tokenizer.encode(resp, add_special_tokens=False)
                    len_tokens = len(tokens)
                    all_response_token_counts.append(len_tokens)
                    traj_tokens += len_tokens
                all_traj_tokens.append(traj_tokens)
            
            avg_response_length = sum(all_response_token_counts) / len(all_response_token_counts) if all_response_token_counts else 0.0
            sample_response_lengths.append(avg_response_length)
            
            avg_num_steps = sum(all_traj_steps) / len(all_traj_steps) if all_traj_steps else 0.0
            avg_traj_tokens = sum(all_traj_tokens) / len(all_traj_tokens) if all_traj_tokens else 0.0
            sample_avg_steps.append(avg_num_steps)
            sample_avg_traj_tokens.append(avg_traj_tokens)
            
            print(f"heldout_rewards_all: {heldout_rewards_all}")
            print(f"[Validation TextGame eval_wo_experience] Held-out Envs Acc: {acc}")
            print(f"[Validation TextGame eval_wo_experience] Avg Response Token Length: {avg_response_length:.2f}")

            # Randomly select and save samples
            env_trajectories = heldout_gen_output.get("env_trajectories", {})
            num_trajectories = len(env_trajectories)
            num_samples_to_save = min(20, num_trajectories)
            sample_indices = random.sample(range(num_trajectories), num_samples_to_save) if num_trajectories > 0 else []
            saved_samples = []
            for idx in sample_indices:
                traj = env_trajectories.get(idx, {})
                history = traj.get("history", [])
                # Build prompt and response from trajectory history
                prompt_parts = []
                response_parts = []
                for step_info in history:
                    prompt_parts.append(f"Step {step_info['step']}: {step_info.get('prompt_text', '')}")
                    response_parts.append(f"Step {step_info['step']}: {step_info.get('raw_response', '')}")
                saved_samples.append({
                    "prompt": "\n".join(prompt_parts),
                    "response": "\n".join(response_parts),
                    "reward": traj.get("reward", 0),
                    "stop_reason": traj.get("stop_reason", "unknown")
                })
            
            # Fixed samples: first 3
            fixed_samples = []
            for idx in range(min(3, num_trajectories)):
                traj = env_trajectories.get(idx, {})
                history = traj.get("history", [])
                prompt_parts = []
                response_parts = []
                for step_info in history:
                    prompt_parts.append(f"Step {step_info['step']}: {step_info.get('prompt_text', '')}")
                    response_parts.append(f"Step {step_info['step']}: {step_info.get('raw_response', '')}")
                fixed_samples.append({
                    "index": idx,
                    "prompt": "\n".join(prompt_parts),
                    "response": "\n".join(response_parts),
                    "reward": traj.get("reward", 0),
                    "stop_reason": traj.get("stop_reason", "unknown")
                })

            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                os.makedirs(val_data_dir, exist_ok=True)
                
                # Helper to write if allowed
                def safe_write_json(filename, data, indent=None):
                    filepath = os.path.join(val_data_dir, filename)
                    if eval_override or not os.path.exists(filepath):
                        with open(filepath, "w") as f:
                            if indent:
                                json.dump(data, f, ensure_ascii=False, indent=indent)
                            else:
                                json.dump(data, f)
                                
                safe_write_json("scores.json", sample_scores)
                safe_write_json("response_lengths.json", sample_response_lengths)
                safe_write_json("avg_steps.json", sample_avg_steps)
                safe_write_json("avg_traj_tokens.json", sample_avg_traj_tokens)
                safe_write_json("random_samples.json", saved_samples, indent=2)
                safe_write_json("fixed_samples.json", fixed_samples, indent=2)
                
                print(f"[Validation TextGame eval_wo_experience] Finished saving results to {val_data_dir}")

            val_metrics = {
                "held_out_acc": acc,
                "avg_response_length": avg_response_length,
                "avg_num_steps": avg_num_steps,
                "avg_traj_tokens": avg_traj_tokens,
                "history_acc": sample_scores,
            }
            return val_metrics

        consumed_samples = 0
        limit = val_samples_limit
        while consumed_samples < limit:
            current_training_seed = training_seeds[consumed_samples]
            consumed_samples += 1
            heldout_rewards_all = []

            for rr in range(HELD_OUT_ROLLOUT):
                # Seed offset to avoid identical stochasticity across rollouts
                rr_seeds = [s + 1000003 * rr for s in heldout_seeds]
                
                heldout_gen_output = self.actor_rollout_wg.generate_sequences_textgame(
                    env_config=self.textgame_env_config,
                    env_num=HELD_OUT_SIZE,                      # per your requirement: no chunking
                    tokenizer=self.tokenizer,
                    experiences=[EXPERIENCE] * HELD_OUT_SIZE,
                    num_steps=num_steps,
                    seeds=rr_seeds,
                    validate=True
                )
                
                if isinstance(heldout_gen_output, list):
                    heldout_gen_output = heldout_gen_output[0]

                heldout_rewards = [reward_dict[0] for reward_dict in heldout_gen_output["reward_list"]]
                heldout_rewards_all.extend(heldout_rewards)

            heldout_reward_tensor = torch.tensor(heldout_rewards_all, dtype=torch.float32)
            acc = (heldout_reward_tensor == 1.0).float().mean().item()
            sample_scores.append(acc)
            print(f"heldout_rewards_all: {heldout_rewards_all}")

            print(f"[Validation Step {consumed_samples}] Held-out Envs Acc: {acc}")
            print(f"History acc: {sample_scores}")
            
            # Calculate metrics
            all_response_token_counts = []
            all_traj_steps = []
            all_traj_tokens = []
            env_trajectories = heldout_gen_output.get("env_trajectories", {})
            for traj in env_trajectories.values():
                history = traj.get("history", [])
                all_traj_steps.append(len(history))
                traj_tokens = 0
                for step_info in history:
                    resp = step_info.get("raw_response", "")
                    tokens = self.tokenizer.encode(resp, add_special_tokens=False)
                    len_tokens = len(tokens)
                    all_response_token_counts.append(len_tokens)
                    traj_tokens += len_tokens
                all_traj_tokens.append(traj_tokens)
            
            avg_response_length = sum(all_response_token_counts) / len(all_response_token_counts) if all_response_token_counts else 0.0
            sample_response_lengths.append(avg_response_length)
            
            avg_num_steps = sum(all_traj_steps) / len(all_traj_steps) if all_traj_steps else 0.0
            sample_avg_steps.append(avg_num_steps)
            
            avg_traj_tokens = sum(all_traj_tokens) / len(all_traj_tokens) if all_traj_tokens else 0.0
            sample_avg_traj_tokens.append(avg_traj_tokens)
            
            print(f"[Validation Step {consumed_samples}] Avg Response Token Length: {avg_response_length:.2f}, Avg Steps: {avg_num_steps:.2f}, Avg Traj Tokens: {avg_traj_tokens:.2f}")

            # Save a random sample from heldout testing
            if env_trajectories:
                traj_idx = random.choice(list(env_trajectories.keys()))
                traj = env_trajectories[traj_idx]
                history = traj.get("history", [])
                prompt_parts = []
                response_parts = []
                for step_info in history:
                    prompt_parts.append(f"Step {step_info['step']}: {step_info.get('prompt_text', '')}")
                    response_parts.append(f"Step {step_info['step']}: {step_info.get('raw_response', '')}")
                val_heldout_samples.append({
                    "step": consumed_samples,
                    "prompt": "\n".join(prompt_parts),
                    "response": "\n".join(response_parts),
                    "reward": traj.get("reward", 0),
                    "stop_reason": traj.get("stop_reason", "unknown")
                })
            
            # Save fixed samples for this step
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                fixed_samples_dir = os.path.join(val_data_dir, "fixed_samples")
                os.makedirs(fixed_samples_dir, exist_ok=True)
                fixed_samples_step = []
                for idx in range(min(3, len(env_trajectories))):
                    traj = env_trajectories.get(idx, {})
                    history = traj.get("history", [])
                    prompt_parts = []
                    response_parts = []
                    for step_info in history:
                        prompt_parts.append(f"Step {step_info['step']}: {step_info.get('prompt_text', '')}")
                        response_parts.append(f"Step {step_info['step']}: {step_info.get('raw_response', '')}")
                    fixed_samples_step.append({
                        "index": idx,
                        "prompt": "\n".join(prompt_parts),
                        "response": "\n".join(response_parts),
                        "reward": traj.get("reward", 0),
                        "stop_reason": traj.get("stop_reason", "unknown")
                    })
                
                fs_path = os.path.join(fixed_samples_dir, f"fixed_samples_{consumed_samples}.json")
                if eval_override or not os.path.exists(fs_path):
                    with open(fs_path, "w") as f:
                        json.dump(fixed_samples_step, f, ensure_ascii=False, indent=2)

            exp_gen_seeds = [current_training_seed]
            
            # NOTE: Currently we always use empty for exp_gen_exp
            exp_gen_experiences = [""]

            exp_gen_output = self.actor_rollout_wg.generate_sequences_textgame(
                env_config=self.textgame_env_config,
                env_num=self.actor_rollout_wg.world_size,
                tokenizer=self.tokenizer,
                experiences=exp_gen_experiences * self.actor_rollout_wg.world_size,
                num_steps=num_steps,
                seeds=exp_gen_seeds * self.actor_rollout_wg.world_size,
                validate=True
            )
            if isinstance(exp_gen_output, list):
                exp_gen_output = exp_gen_output[0]

            exp_traj = exp_gen_output["env_trajectories"][0]
            assert len(exp_traj.get("history", [])) > 0, "Experience env has empty history"

            # Format multi-round history for exp learner
            multi_full_history = ""
            for step_info in exp_traj["history"]:
                step_num = step_info["step"]
                obs = step_info["current_step_observation"]
                if step_num == 0 and "Sokoban" in obs:
                    obs = obs.replace("You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets.\n        When you are right next to a box, you can push it by moving in the same direction.\n        You cannot push a box through a wall, and you cannot pull a box.\n        On the board, objects are represented as: \n        - The player (you) appears as 'P' \n        - Walls are represented with '#' \n        - Boxes are marked as 'X' \n        - Empty goals are shown with a 'O'\n        - Boxes on goals are visualized with '√'\n        You can also use [w] for up, [a] for left, [s] for down, and [d] for right.", "").strip()
                elif step_num == 0 and "Frozen Lake" in obs:
                    obs = obs.replace("Welcome to Frozen Lake!\n\nYou are represented by 'P' on the grid.\nGrid symbols:\n  ' ' = Frozen surface (safe to walk on)\n  'H' = Hole (fall in and lose!)\n  'G' = Goal (reach this to win!)\n  'P' = Your current position\n\nAvailable actions: up, down, left, right (or w, a, s, d)\nType your action as: [up], [down], [left], [right] or [w], [a], [s], [d]\n\nObjective: Navigate from the start (top-left) to the goal (bottom-right) without falling into any holes!\n\n", "").strip()
                raw_response = step_info["raw_response"]
                multi_full_history += f"\nRound{step_num}_Input: {obs}\n\nRound{step_num}_Output: {raw_response}"
            
            if self.config.trainer.textgame_wfeedback:
                multi_full_history += f"\n\n\n{exp_traj.get('final_feedback', '')}\n"

            latest_experience = multi_full_history
            previous_experience = EXPERIENCE if EXPERIENCE else "No previous experience."

            if not self.config.trainer.exp_sel_with_prev:
                exp_learner_prompt = EXPERIENCE_UPDATE_PROMPT.format(
                    PREVIOUS_EXPERIENCE="No previous experience.",
                    LATEST_EXPERIENCE=latest_experience,
                )
            else:
                exp_learner_prompt = EXPERIENCE_UPDATE_PROMPT.format(
                    PREVIOUS_EXPERIENCE=previous_experience,
                    LATEST_EXPERIENCE=latest_experience,
                )
            exp_learner_msgs = [{"role": "user", "content": exp_learner_prompt}]


            prompt_with_template = self.tokenizer.apply_chat_template(
                exp_learner_msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=not self.textgame_env_config["no_think"],
            )
            exp_learner_tokenized = self.tokenizer(
                prompt_with_template,
                return_tensors="pt",
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.config.data["max_prompt_length"],
            )
            exp_learner_tokenized["position_ids"] = compute_position_id_with_mask(
                exp_learner_tokenized["attention_mask"]
            )

            # Build DataProto for exp_learner
            exp_learner_batch = DataProto.from_single_dict({
                "input_ids": exp_learner_tokenized["input_ids"],
                "attention_mask": exp_learner_tokenized["attention_mask"],
                "position_ids": exp_learner_tokenized["position_ids"],
            })
            exp_learner_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "n": 1,
            }
            exp_learner_batch_padded, exp_pad_size = pad_dataproto_to_divisor(
                exp_learner_batch,
                self.exp_learner_wg.world_size
            )
            exp_learner_output_padded = self.exp_learner_wg.generate_sequences(exp_learner_batch_padded)
            exp_learner_output = unpad_dataproto(exp_learner_output_padded, exp_pad_size * 1)

            exp_text = self.tokenizer.decode(
                exp_learner_output.batch["responses"][0],
                skip_special_tokens=True
            )
            if "</think>" in exp_text:
                exp_text = exp_text.split("</think>")[-1]

            # Parse experience lines based on prompt version
            if self.config.trainer.prompt_version == 'v2':
                # v2: use simple markdown bullet points
                exp_lines = exp_text.split("\n")
                exp_lines = [line.strip() for line in exp_lines if line.strip().startswith("- ")]
                parsed_exp_text = "\n".join(exp_lines).strip()
            elif self.config.trainer.prompt_version in ['v1', 'v4']:
                # v1 and v4: use - EXPERIENCE ITEM: marker
                marker = "- EXPERIENCE ITEM:"
                exp_lines = exp_text.split("\n")
                exp_lines = [line for line in exp_lines if marker in line]

                result_lines = []
                for line in exp_lines:
                    parts = line.split(marker)[1:]  # skip first empty
                    result_lines.extend([marker + p.rstrip() for p in parts if p.strip()])

                parsed_exp_text = "\n".join(result_lines).strip()
            elif self.config.trainer.prompt_version in ['v3', 'v5']:
                parsed_exp_text = exp_text.strip()
            
            if EXPERIENCE and EXPERIENCE != "No previous experience." and parsed_exp_text:
                EXPERIENCE = EXPERIENCE + "\n" + parsed_exp_text
            elif parsed_exp_text:
                EXPERIENCE = parsed_exp_text

            # Truncate experience if too long
            max_exp_tokens = self.config.trainer.experience_max_length
            EXPERIENCE = self._truncate_experience(EXPERIENCE, max_exp_tokens)

            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                exp_dir = os.path.join(val_data_dir, "experiences")
                os.makedirs(exp_dir, exist_ok=True)
                with open(os.path.join(exp_dir, f"experience_{consumed_samples}.txt"), "w") as f:
                    f.write(EXPERIENCE)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            os.makedirs(val_data_dir, exist_ok=True)
            
            # Helper to write if allowed
            def safe_write_json(filename, data, indent=None):
                filepath = os.path.join(val_data_dir, filename)
                if eval_override or not os.path.exists(filepath):
                    with open(filepath, "w") as f:
                        if indent:
                            json.dump(data, f, ensure_ascii=False, indent=indent)
                        else:
                            json.dump(data, f)
            
            safe_write_json("scores.json", sample_scores)
            safe_write_json("response_lengths.json", sample_response_lengths)
            safe_write_json("avg_steps.json", sample_avg_steps)
            safe_write_json("avg_traj_tokens.json", sample_avg_traj_tokens)
            safe_write_json(f"{val_samples_limit}.json", val_heldout_samples, indent=2)
                
            print(f"[Validation TextGame] Finished saving results to {val_data_dir}")

        val_metrics = {
            "held_out_acc": acc,
            "avg_response_length": avg_response_length,
            "avg_num_steps": avg_num_steps,
            "avg_traj_tokens": avg_traj_tokens,
            "history_acc": sample_scores,
        }
        return val_metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            ) # NOTE: we can change role to rollout later if we don't train it 
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

            # Skip creating exp_learner when stage is "consolidate"
            if self.config.trainer.stage != "consolidate":
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ExpLearner)
                exp_config = deepcopy(self.config.actor_rollout_ref)
                if OmegaConf.select(exp_config.model, "exp_model_path", default=None) is not None:
                    exp_config.model.path = exp_config.model.exp_model_path
                exp_learner_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ExpLearner],
                    config=exp_config,
                    role="exp_learner",
                )
                self.resource_pool_to_cls[resource_pool]["exp_learner"] = exp_learner_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_config = deepcopy(self.config.actor_rollout_ref)
            if OmegaConf.select(ref_config.model, "ref_model_path", default=None) is not None:
                ref_config.model.path = ref_config.model.ref_model_path
                print("Using RefModel: ", ref_config.model.path)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=ref_config, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        # Skip initializing exp_learner when stage is "consolidate"
        if self.config.trainer.stage != "consolidate":
            self.exp_learner_wg = all_wg["exp_learner"]
            self.exp_learner_wg.init_model()
        else:
            self.exp_learner_wg = None

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_exp_learner_ckpt_to_keep = self.config.trainer.get("max_exp_learner_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        if self.config.trainer.stage == "consolidate":
            self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.config.trainer.stage != "consolidate":
            exp_learner_local_path = os.path.join(local_global_step_folder, "exp_learner")
            exp_learner_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "exp_learner")
            self.exp_learner_wg.save_checkpoint(exp_learner_local_path, exp_learner_remote_path, self.global_steps, max_ckpt_to_keep=max_exp_learner_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        if not self.is_textgame and getattr(self, "train_dataloader", None) is not None:
            BaseCheckpointManager.local_mkdir(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        exp_learner_path = os.path.join(global_step_folder, "exp_learner")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load exp learner
        if self.config.trainer.stage != "consolidate":
            self.exp_learner_wg.load_checkpoint(exp_learner_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        if not self.is_textgame and getattr(self, "train_dataloader", None) is not None:
            # TODO: from remote not implemented yet
            dataloader_local_path = os.path.join(global_step_folder, "data.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            else:
                print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        import torch
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if not self.is_textgame:
            if self.config.trainer.prompt_version == 'v3':
                EXPERIENCE_UPDATE_PROMPT = """
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction (the user's question and your answer):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning. The experience you generate will be directly appended to the previous experience.

After careful reasoning step by step, output the final additional experience.
"""

            elif self.config.trainer.prompt_version == 'v4':
                EXPERIENCE_UPDATE_PROMPT = """
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction (the user's question and your answer):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning.

Rules:
- The experience you generate MUST be formatted strictly as a markdown list where each item starts with "- EXPERIENCE ITEM:", one per line:
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- The experience you generate will be directly appended to the previous experience.
- The change should introduce a general, high-level, widely applicable insight, not a detail from the specific interaction. The updated experience must remain concise, structured, and meaningful.
- If the new insight conflicts with any previous experience item, you are can describe the conflict and provide a resolution in the new item.

After careful reasoning step by step, output the final result in exactly this format:

Additional Experience:
# Experience
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
"""


        else:
            if self.config.trainer.prompt_version == 'v3':
                EXPERIENCE_UPDATE_PROMPT = """You are an AI language model that continuously refines its internal experience.
Here is the interaction history (the game environment (input) and your response and action (output)):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the multi-round interaction history and the previous experience, generate experience for future learning. You should conduct a deep, comparative analysis to infer the game rules and the fundamental principles behind winning and losing. Using the interaction history and environment feedback, hypothesize the game rules and effective winning strategies, and organize these insights into experience items that help the player succeed in the game.

Rules:
- The experience you generate will be directly appended to the previous experience. Do not repeat the previous experience. Make sure the newly generated experience is different from the previous experience.
- Your generated experience should be possible rules, instructions or winning strategies for the game. The experience should be generally useful rather than only applicable for the current map (board).

After careful reasoning step by step, output the final additional experience.
"""
            elif self.config.trainer.prompt_version == 'v4':
                EXPERIENCE_UPDATE_PROMPT = """You are an AI language model that continuously refines its internal experience.
Here is the interaction history (the game environment (input) and your response and action (output)):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the multi-round interaction history and the previous experience, generate experience for future learning. You should conduct a deep, comparative analysis to infer the game rules and the fundamental principles behind winning and losing. Using the interaction history and environment feedback, hypothesize the game rules and effective winning strategies, and organize these insights into 1-2 concise, high-level, and widely applicable experience items that help the player succeed in the game.

Rules:
- The experience you generate MUST be formatted strictly as a markdown item which starts with "- EXPERIENCE ITEM:":
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- The experience you generate will be directly appended to the previous experience. Do not repeat the previous experience. Make sure the newly generated experience is different from the previous experience.
- Your generated experience should be possible rules, instructions or winning strategies for the game. The experience should be generally useful rather than only applicable for the current map (board).

After careful reasoning step by step, output the final result in exactly this format:

Additional Experience (Rules or Strategies):
# Experience
- EXPERIENCE ITEM: ...
"""
        
        EXPERIENCE_SOLVE_PROMPT_TEMPLATE = """Given previous learned experience:
# Experience
{experience}

Solve the new problem and explain what part of experience you use and how you use it in the reasoning process:
{prompt}"""


        self.experience_update_prompt = EXPERIENCE_UPDATE_PROMPT
        self.experience_solve_prompt_template = EXPERIENCE_SOLVE_PROMPT_TEMPLATE

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        EXPERIENCE = "No previous experience."
        self.experience = EXPERIENCE
        
        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter([None]) if self.is_textgame else self.train_dataloader
            for batch_dict in dataloader_iter:
                
                if self.config.trainer.stage == "consolidate":
                    do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                    if do_profile:
                        self.actor_rollout_wg.start_profile()
                        self.exp_learner_wg.start_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                    metrics = {}
                    timing_raw = {}
                    
                    assert self.is_textgame
                    # textgame get seed number using current training step to make at same step, the training data is same. Get length of seed using batchsize in config. No batch and gen_batch now
                    env_num = self.config.data.train_batch_size
                    seeds = [505019424 + 90039 + 100000 + (self.config.trainer.oel_round - 1) * 10000000 + tmp_num * 1000 for tmp_num in range(self.global_steps * env_num, (self.global_steps + 1) * env_num)]
                    num_steps = self.config.trainer.textgame_max_steps        
                    
                    batch = None
                    gen_batch = None
                    
                    is_last_step = self.global_steps >= self.total_training_steps
                    
                    exp_path = self.config.trainer.experience_path
                    
                    multi_experience = self.config.trainer.multi_experience
                    EXPERIENCES = []
                    
                    if not os.path.exists(exp_path):
                        print("[Experience Load] Experience not found, using empty string")
                        EXPERIENCES = ['']
                    else:
                        if multi_experience:
                            with open(exp_path, "r", encoding="utf-8") as f:
                                exp_paths = [line.strip() for line in f if line.strip()]
                            for p in exp_paths:
                                with open(p, "r", encoding="utf-8") as f:
                                    content = f.read().strip()
                                max_exp_tokens = self.config.trainer.experience_max_length
                                content = self._truncate_experience(content, max_exp_tokens)
                                EXPERIENCES.append(content)
                            print(f"[Experience Load] Loaded {len(EXPERIENCES)} experiences from {exp_path}")
                        else:
                            with open(exp_path, "r", encoding="utf-8") as f:
                                EXPERIENCE = f.read().strip()
                            max_exp_tokens = self.config.trainer.experience_max_length
                            EXPERIENCE = self._truncate_experience(EXPERIENCE, max_exp_tokens)
                            EXPERIENCES = [EXPERIENCE]
                            print(f"[Experience Load] Loaded experience from {exp_path}")
                    
                    EXPERIENCE = EXPERIENCES[0]
                    self.experience = EXPERIENCES if multi_experience else EXPERIENCES[0]
                    
                    # Fix random seed for experience sampling
                    sampling_seed = self.global_steps + self.config.trainer.oel_round * 1000
                    rng = random.Random(sampling_seed)

                    with marked_timer("step", timing_raw):
                        with marked_timer("gen", timing_raw, color="red"):
                            assert self.config.trainer.train_oel
                            
                            if self.config.trainer.on_policy_merge:
                                exp_name = self.config.trainer.experiment_name
                                train_step = self.global_steps
                                load_dir = self.config.trainer.deploy_save_dir
                                load_path = os.path.join(load_dir, f"gen_batch_step_{train_step}.pkl")
                                assert os.path.exists(load_path)
                                with open(load_path, "rb") as f:
                                    gen_batch = pickle.load(f)
                                print(f"[Off-Policy Load] Loaded gen_batch from {load_path}")
                                
                                # gen_batch should be multiple of self.actor_rollout_wg.world_size, we drop some items here
                                gen_batch = gen_batch[:len(gen_batch) - len(gen_batch) % self.actor_rollout_wg.world_size]

                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                
                            else:
                                gen_output = self.actor_rollout_wg.generate_sequences_textgame(
                                    env_config=self.textgame_env_config,
                                    env_num=env_num,
                                    tokenizer=self.tokenizer,
                                    experiences=[""] * env_num,
                                    num_steps=num_steps,
                                    seeds=seeds
                                )
                            
                                if isinstance(gen_output, list):
                                    gen_output = gen_output[0]
                                    
                                env_trajectories = gen_output['env_trajectories']
                                
                                prompt_token_ids_list = []
                                response_ids_list = []
                                raw_prompt_list = []
                                for env_idx, traj in env_trajectories.items():
                                    history = traj['history']
                                    for step_item in history:
                                        prompt_token_ids = step_item['prompt_token_ids']
                                        response_ids = step_item['response_ids']
                                        raw_prompt = step_item['message']
                                        
                                        assert len(prompt_token_ids) <= self.config.data.max_prompt_length
                                        assert len(response_ids) > 0
                                            
                                        prompt_token_ids_list.append(prompt_token_ids)
                                        response_ids_list.append(response_ids)
                                        raw_prompt_list.append(raw_prompt)

                                gen_batch = self._build_dataproto_woresp(prompt_token_ids_list)
                                gen_batch.non_tensor_batch['raw_prompt'] = np.array(raw_prompt_list, dtype=object)

                                exp_name = self.config.trainer.experiment_name
                                train_step = self.global_steps
                                save_dir = self.config.trainer.deploy_save_dir
                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, f"gen_batch_step_{train_step}.pkl")
                                with open(save_path, "wb") as f:
                                    pickle.dump(gen_batch, f)
                                print(f"[Off-Policy Save] Saved gen_batch to {save_path}")

                                is_last_step = self.global_steps >= self.total_training_steps
                                self.global_steps += 1
                                if is_last_step:
                                    progress_bar.close()
                                    return
                                continue

                            batch = gen_batch_output.select(deepcopy=True)
                            batch_with_exp = None
                                          
                            gen_batch_with_exp = batch.select(deepcopy=True)
                            
                            updated_gen_inputs = []
                            for i in range(len(gen_batch_with_exp)):
                                msgs = deepcopy(gen_batch_with_exp.non_tensor_batch['raw_prompt'][i])
                                content = msgs[-1]['content']
                                exp_to_use = rng.choice(EXPERIENCES)
                                if exp_to_use:
                                    remove_str = "You are an agent playing a game on a grid, acting as a reasoning engine.\n\nCurrent situation:\n"
                                    add_str = f"You are an agent playing a game on a grid, acting as a reasoning engine.\n\nYour decisions are based on the experience you have learned about the game's rules or strategies. This experience is only a guess of how the game works, and the rules and strategies may be incomplete or incorrect.\n\nGiven experience for rules or strategies you have learned:\n{exp_to_use}\n\nCurrent situation:\n"
                                    updated_content = add_str + content[len(remove_str):]
                                else:
                                    updated_content = content
                                msgs[-1]['content'] = updated_content
                                tokenized = self._re_tokenize(msgs)
                                updated_gen_inputs.append(tokenized)
                            
                            gen_batch_with_exp.batch["input_ids"] = torch.stack([inp["input_ids"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.batch["attention_mask"] = torch.stack([inp["attention_mask"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.batch["position_ids"] = torch.stack([inp["position_ids"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.non_tensor_batch.pop("raw_prompt_ids", None)
                            gen_batch_with_exp.non_tensor_batch.pop("raw_prompt", None)

                            batch_with_exp = gen_batch_with_exp.select(deepcopy=True)

                        assert self.config.trainer.on_policy_merge
                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        
                        # Construct batch_with_exp: use gen_batch_with_exp's prompts + gen_batch_output's responses
                        # First repeat gen_batch_with_exp and batch_with_exp to match rollout.n
                        batch_with_exp.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_with_exp.batch))], dtype=object)
                        batch_with_exp = batch_with_exp.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        gen_batch_with_exp = gen_batch_with_exp.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        
                        # Get prompts from gen_batch_with_exp and responses from gen_batch_output
                        prompts_with_exp = gen_batch_with_exp.batch["input_ids"]  # (bs, prompt_length)
                        prompt_attention_mask = gen_batch_with_exp.batch["attention_mask"]
                        prompt_position_ids = gen_batch_with_exp.batch["position_ids"]
                        responses = gen_batch_output.batch["responses"]  # (bs, response_length)
                        
                        # Concatenate prompts and responses to form the full sequence
                        seq_with_exp = torch.cat([prompts_with_exp, responses], dim=-1)
                        
                        # Build response attention mask (following vllm_rollout_spmd.py logic)
                        eos_token_id = batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
                        response_attention_mask = get_response_mask(response_id=responses, eos_token=eos_token_id, dtype=prompt_attention_mask.dtype)
                        attention_mask_with_exp = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
                        
                        # Build response position ids
                        response_length = responses.size(1)
                        delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
                        delta_position_id = delta_position_id.unsqueeze(0).expand(prompts_with_exp.size(0), -1)
                        if prompt_position_ids.dim() == 3:  # qwen2vl mrope
                            delta_position_id = delta_position_id.view(prompts_with_exp.size(0), 1, -1).expand(prompts_with_exp.size(0), 3, -1)
                        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
                        position_ids_with_exp = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
                        
                        # Update batch_with_exp with the new tensors
                        batch_with_exp.batch["prompts"] = prompts_with_exp
                        batch_with_exp.batch["responses"] = responses
                        batch_with_exp.batch["input_ids"] = seq_with_exp
                        batch_with_exp.batch["attention_mask"] = attention_mask_with_exp
                        batch_with_exp.batch["position_ids"] = position_ids_with_exp
                        batch_with_exp.batch["response_mask"] = compute_response_mask(batch_with_exp)
                        
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        if not self.config.trainer.skip_reward:
                            with marked_timer("reward", timing_raw, color="yellow"):
                                gen_output_rewards = torch.tensor(gen_output_rewards, dtype=torch.float32)
                                acc = (gen_output_rewards == 1.0).float().mean().item()
                                metrics.update({
                                    "actor/curr_acc": acc
                                })
                                
                        # get topk logits indices
                        if self.config.actor_rollout_ref.actor.kl_loss_type == "full" and self.config.actor_rollout_ref.actor.kl_topk > 0:
                            batch.meta_info["return_all_logits"] = True
                            batch_with_exp.meta_info["return_all_logits"] = True
                            with marked_timer("compute_topk_indices", timing_raw, color="purple"):
                                log_prob_proto = self.actor_rollout_wg.compute_log_prob(batch)
                                log_probs = log_prob_proto.batch["old_log_probs"]
                                
                                actor_topk_indices = log_probs.long()
                                
                                if self.config.actor_rollout_ref.actor.kl_merge_indice:
                                    batch_with_exp.batch["first_kl_topk_indices"] = actor_topk_indices
                                    ref_log_prob_for_topk = self.ref_policy_wg.compute_ref_log_prob(batch_with_exp)
                                    merged_topk_indices = ref_log_prob_for_topk.batch["ref_log_prob"].long()
                                    batch_with_exp.batch["kl_topk_indices"] = merged_topk_indices
                                    del ref_log_prob_for_topk
                                    del merged_topk_indices
                                else:
                                    batch_with_exp.batch["kl_topk_indices"] = actor_topk_indices

                                del log_probs
                                del log_prob_proto
                                    
                                    
                        # use batch_with_exp to compute exp_log_prob
                        if self.config.actor_rollout_ref.actor.kl_loss_type != "seqkd":
                            with marked_timer("exp_log_prob", timing_raw, color="olive"):
                                # For topk, we set return_all_logits=True so dp_actor triggers the topk logic block.
                                # But since kl_topk_indices is now in batch, dp_actor will perform Step 2 (gathering).
                                # see _forward_micro_batch in verl/verl/workers/actor/dp_actor.py
                                batch_with_exp.meta_info["return_all_logits"] = self.config.actor_rollout_ref.actor.kl_loss_type == "full"
                                assert self.use_reference_policy
                                exp_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch_with_exp)

                                exp_entropys = exp_log_prob.batch["entropys"]
                                response_masks = batch_with_exp.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(loss_mat=exp_entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                exp_prob_metrics = {"actor/exp_entropy": entropy_agg.detach().item()}
                                metrics.update(exp_prob_metrics)
                                exp_log_prob.batch.pop("entropys")
                                exp_log_prob.batch["exp_log_probs"] = exp_log_prob.batch["ref_log_prob"]
                                exp_log_prob.batch.pop("ref_log_prob")
                                
                                if self.config.actor_rollout_ref.actor.kl_loss_type == "full" and self.config.actor_rollout_ref.actor.kl_topk > 0:
                                    exp_log_prob.batch["kl_topk_indices"] = batch_with_exp.batch["kl_topk_indices"]

                                batch = batch.union(exp_log_prob)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["stage_merge"] = True
                                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                batch.meta_info["on_policy_merge"] = self.config.trainer.on_policy_merge
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                            
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                                print(batch.batch.keys())
                                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )

                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    if not self.is_textgame:
                        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                        # TODO: implement actual tflpo and theoretical tflpo
                        n_gpus = self.resource_pool_manager.get_n_gpus()
                        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                
                
                else:
                    raise ValueError(f"Unknown trainer stage: {self.config.trainer.stage}")