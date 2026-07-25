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
import re
import uuid
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
)
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask

from verl.utils.torch_functional import get_response_mask
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
    RefPolicyStatic = 8


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

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

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

        jsd_beta = config.actor_rollout_ref.actor.get("jsd_beta", -1)
        if jsd_beta > 0:
            if not jsd_beta < 1:
                raise ValueError(f"jsd_beta must be between 0 and 1, got {jsd_beta}")
            if config.actor_rollout_ref.actor.kl_loss_type != "full":
                raise ValueError("JSD requires actor_rollout_ref.actor.kl_loss_type=full")

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

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        # Check for multi-dataset mode
        multidata_ratio = self.config.data.get("multidata_ratio", None)
        multidata_rm_prompt = self.config.trainer.get("multidata_rm_prompt", None)
        train_files = self.config.data.train_files
        if isinstance(train_files, str):
            train_files = [train_files]
        elif OmegaConf.is_list(train_files):
            train_files = list(train_files)

        if multidata_ratio is not None and len(train_files) > 1:
            if OmegaConf.is_list(multidata_ratio):
                multidata_ratio = list(multidata_ratio)
            assert len(multidata_ratio) == len(train_files), f"multidata_ratio length {len(multidata_ratio)} != train_files length {len(train_files)}"
            if multidata_rm_prompt is not None:
                if OmegaConf.is_list(multidata_rm_prompt):
                    multidata_rm_prompt = list(multidata_rm_prompt)
                assert len(multidata_rm_prompt) == len(train_files), f"multidata_rm_prompt length {len(multidata_rm_prompt)} != train_files length {len(train_files)}"

            self.multidata_ratio = [float(r) for r in multidata_ratio]
            self.multidata_rm_prompt = multidata_rm_prompt
            base_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)

            self.train_dataloaders = []
            for i, (f, ratio) in enumerate(zip(train_files, self.multidata_ratio)):
                ds = create_rl_dataset(f, self.config.data, self.tokenizer, self.processor)
                sampler = create_rl_sampler(self.config.data, ds)
                bs = int(base_batch_size * ratio)
                dl = StatefulDataLoader(
                    dataset=ds, batch_size=bs,
                    num_workers=self.config.data.get("dataloader_num_workers", 8),
                    drop_last=True, collate_fn=collate_fn, sampler=sampler,
                )
                self.train_dataloaders.append(dl)
                print(f"[MultiData] Dataset {i}: {f}, batch_size={bs}, ratio={ratio}, len={len(dl)}")

            self.train_dataloader = None
            self.train_dataset = self.train_dataloaders[0].dataset
        else:
            self.train_dataloaders = None
            self.multidata_ratio = None
            self.multidata_rm_prompt = None
            if train_dataset is None:
                train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
            self.train_dataset = train_dataset

            if train_sampler is None:
                train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )

        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.val_dataset = val_dataset

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

        if self.train_dataloader is not None:
            assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
            self.steps_per_epoch = len(self.train_dataloader)
            total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        else:
            min_dl_len = min(len(dl) for dl in self.train_dataloaders)
            print(f"[MultiData] Min dataloader length: {min_dl_len}, Size of val dataloader: {len(self.val_dataloader)}")
            self.steps_per_epoch = min_dl_len
            total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

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

    def _concat_batch_dicts(self, batch_dicts):
        """Tag per-sample rm_prompt_version and concat batch_dicts from multiple dataloaders."""
        multidata_fixed_teacher = self.config.trainer.get("multidata_fixed_teacher", None)
        if multidata_fixed_teacher is not None:
            if OmegaConf.is_list(multidata_fixed_teacher):
                multidata_fixed_teacher = list(multidata_fixed_teacher)
        for i, bd in enumerate(batch_dicts):
            rm_ver = self.multidata_rm_prompt[i] if self.multidata_rm_prompt is not None else self.config.trainer.rm_prompt_version
            bs = bd["input_ids"].shape[0]
            bd["rm_prompt_version"] = np.array([rm_ver] * bs, dtype=object)
            if multidata_fixed_teacher is not None:
                bd["use_fixed_teacher"] = np.array([int(multidata_fixed_teacher[i])] * bs)
        combined = {}
        for key in batch_dicts[0].keys():
            vals = [bd[key] for bd in batch_dicts]
            if isinstance(vals[0], torch.Tensor):
                combined[key] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], np.ndarray):
                combined[key] = np.concatenate(vals, axis=0)
            elif isinstance(vals[0], list):
                combined[key] = sum(vals, [])
            else:
                combined[key] = vals[0]
        return combined

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
        # Lists to collect samples for the table
        sample_scores = []
        sample_response_lengths = []  # Track average response token lengths

        HELD_OUT_SIZE = self.config.trainer.get("held_out_size", 500)
        HELD_OUT_SIZE = min(HELD_OUT_SIZE, len(self.val_dataloader))
        print("HELD_OUT_SIZE: ", HELD_OUT_SIZE)
        HELD_OUT_ROLLOUT = self.config.trainer.get("held_out_rollout", 2)
        val_iterator = iter(self.val_dataloader)
        heldout_data_list = []

        # Collect heldout batch
        for _ in range(HELD_OUT_SIZE):
            try:
                data = next(val_iterator)
                heldout_data_list.append(data)
            except StopIteration:
                break

        print("[Validation] Evaluating held-out prompts without experience context...")
        heldout_gen_list = []
        heldout_meta_list = []

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "raw_prompt"]

        for heldout_data in heldout_data_list:
            hb_meta = DataProto.from_single_dict(heldout_data)

            hb_gen = hb_meta.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            hb_gen.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }


            hb_gen.non_tensor_batch.pop("raw_prompt_ids", None)
            hb_gen.non_tensor_batch.pop("raw_prompt", None)

            heldout_gen_list.append(hb_gen)
            heldout_meta_list.append(hb_meta)

        heldout_gen_batch = DataProto.concat(heldout_gen_list)

        # Pad and Generate
        heldout_gen_batch_padded, pad_size = pad_dataproto_to_divisor(heldout_gen_batch, self.actor_rollout_wg.world_size)
        heldout_gen_batch_padded.meta_info["n"] = HELD_OUT_ROLLOUT

        if not self.async_rollout_mode:
            heldout_output_padded = self.actor_rollout_wg.generate_sequences(heldout_gen_batch_padded)
        else:
            self.async_rollout_manager.wake_up()
            heldout_output_padded = self.async_rollout_manager.generate_sequences(heldout_gen_batch_padded)
            self.async_rollout_manager.sleep()

        heldout_output = unpad_dataproto(heldout_output_padded, pad_size * heldout_gen_batch_padded.meta_info["n"])

        # Calculate average response token length
        responses = heldout_output.batch["responses"]
        response_lengths = (responses != self.tokenizer.pad_token_id).sum(dim=-1).float()
        avg_response_length = response_lengths.mean().item()
        sample_response_lengths.append(avg_response_length)
        print(f"[Validation eval_wo_experience] Avg Response Token Length: {avg_response_length:.2f}")

        # Compute Reward (Metric)
        heldout_meta_batch = DataProto.concat(heldout_meta_list)
        heldout_meta_batch = heldout_meta_batch.repeat(repeat_times=heldout_gen_batch_padded.meta_info["n"], interleave=True)
        eval_batch = heldout_meta_batch.union(heldout_output)

        if self.skip_rm_scoring:
            # Skip RM scoring, just save responses with score=0
            n_responses = responses.shape[0]
            val_rm_scores = [0.0] * n_responses
            val_rm_mean = 0.0
            val_rm_valid_ratio = 0.0
            val_rm_avg_len = 0.0
            sample_scores.append(0.0)
            print(f"[Validation] skip_rm_scoring=True, scores set to 0")
        else:
            # Use exp_learner_wg as reward model to score responses
            n_responses = responses.shape[0]
            rm_prompt_token_ids_list = []
            for i in range(n_responses):
                msgs = eval_batch[i].non_tensor_batch.get('raw_prompt', None)
                if isinstance(msgs, np.ndarray):
                    msgs = msgs.tolist()
                if isinstance(msgs, list):
                    instruction = msgs[-1]['content']
                else:
                    instruction = str(msgs)

                response_text = self.tokenizer.decode(responses[i], skip_special_tokens=True)
                rubric_list_string = eval_batch[i].non_tensor_batch.get('rubric_list_string', '')
                if isinstance(rubric_list_string, np.ndarray):
                    rubric_list_string = str(rubric_list_string)

                rm_prompt = self._rubric_based_prompt_template.format(
                    instruction=instruction,
                    response=response_text,
                    rubric_list_string=rubric_list_string,
                )
                rm_msgs = [{"role": "user", "content": rm_prompt}]
                rm_prompt_with_template = self.tokenizer.apply_chat_template(
                    rm_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                rm_tokenized = self.tokenizer(
                    rm_prompt_with_template, return_tensors="pt", add_special_tokens=False,
                    padding=False, truncation=True, max_length=self.config.data["max_prompt_length"],
                )
                rm_prompt_token_ids_list.append(rm_tokenized["input_ids"][0].tolist())

            rm_batch = self._build_dataproto_woresp(rm_prompt_token_ids_list)
            rm_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "n": 1,
            }
            rm_batch_padded, rm_pad_size = pad_dataproto_to_divisor(rm_batch, self.exp_learner_wg.world_size)
            rm_output_padded = self.exp_learner_wg.generate_sequences(rm_batch_padded)
            rm_output = unpad_dataproto(rm_output_padded, rm_pad_size)

            val_rm_scores = []
            val_rm_valid_count = 0
            val_rm_response_lengths = []
            for i in range(n_responses):
                rm_response_text = self.tokenizer.decode(rm_output.batch["responses"][i], skip_special_tokens=True)
                rm_resp_tokens = rm_output.batch["responses"][i]
                val_rm_response_lengths.append((rm_resp_tokens != self.tokenizer.pad_token_id).sum().item())
                score = 1.0
                score_match = re.search(r'<score>(.*?)</score>', rm_response_text, re.DOTALL | re.IGNORECASE)
                if score_match:
                    score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_match.group(1).strip())
                    if score_number_match:
                        try:
                            score = float(score_number_match.group(1))
                            score = max(1.0, min(10.0, score))
                            val_rm_valid_count += 1
                        except ValueError:
                            score = 1.0
                val_rm_scores.append(score)

            val_rm_mean = sum(val_rm_scores) / len(val_rm_scores) / 10.0
            val_rm_valid_ratio = val_rm_valid_count / n_responses if n_responses > 0 else 0.0
            val_rm_avg_len = sum(val_rm_response_lengths) / len(val_rm_response_lengths) if val_rm_response_lengths else 0.0
            sample_scores.append(val_rm_mean)
            print(f"[Validation] RM score: {val_rm_mean:.4f}, valid_ratio: {val_rm_valid_ratio:.4f}, rm_avg_len: {val_rm_avg_len:.1f}")

        # Save samples
        num_samples_to_save = len(responses)
        sample_indices = list(range(num_samples_to_save))
        saved_samples = []
        for idx in sample_indices:
            prompt_ids = heldout_output.batch["prompts"][idx]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_text = self.tokenizer.decode(responses[idx], skip_special_tokens=True)
            rubric_list_string = eval_batch[idx].non_tensor_batch.get('rubric_list_string', '')
            if isinstance(rubric_list_string, np.ndarray):
                rubric_list_string = str(rubric_list_string)
            sample = {
                "prompt": prompt_text,
                "response": response_text,
                "rm_score": val_rm_scores[idx],
                "rubric_list_string": rubric_list_string,
            }
            saved_samples.append(sample)

        # dump results
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            os.makedirs(val_data_dir, exist_ok=True)
            with open(os.path.join(val_data_dir, "scores.json"), "w") as f:
                json.dump(sample_scores, f)
            with open(os.path.join(val_data_dir, "response_lengths.json"), "w") as f:
                json.dump(sample_response_lengths, f)
            # Save random samples
            with open(os.path.join(val_data_dir, "random_samples.json"), "w") as f:
                json.dump(saved_samples, f, ensure_ascii=False, indent=2)
            print(f"[Validation eval_wo_experience] Saved {len(saved_samples)} random samples to {val_data_dir}/random_samples.json")

        val_result = {
            "held_out_rm_score": val_rm_mean,
            "held_out_rm_valid_ratio": val_rm_valid_ratio,
            "held_out_rm_avg_len": val_rm_avg_len,
        }
        return val_result

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

            exp_model_path_str = str(OmegaConf.select(self.config.actor_rollout_ref.model, "exp_model_path", default="") or "")
            self.use_gpt_rm = "gpt" in exp_model_path_str
            self.skip_rm_scoring = exp_model_path_str.lower() == "none"

            if not self.use_gpt_rm and not self.skip_rm_scoring and (self.config.trainer.stage != "consolidate" or self.config.trainer.use_exp_model):
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

        # create static ref policy (fixed base teacher for multidata_fixed_teacher)
        if Role.RefPolicyStatic in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicyStatic)
            static_ref_config = deepcopy(self.config.actor_rollout_ref)
            if OmegaConf.select(static_ref_config.model, "ref_model_path", default=None) is not None:
                static_ref_config.model.path = static_ref_config.model.ref_model_path
                print("Using Static RefModel: ", static_ref_config.model.path)
            static_ref_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicyStatic], config=static_ref_config, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref_static"] = static_ref_cls

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

        if "ref_static" in all_wg:
            self.ref_policy_static_wg = all_wg["ref_static"]
            self.ref_policy_static_wg.init_model()
        else:
            self.ref_policy_static_wg = None

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        # Create exp_learner when stage is not "consolidate" or when use_exp_model is True
        if self.use_gpt_rm:
            from verl.utils.gpt_rm import GptRMClient
            exp_model_path_str = str(self.config.actor_rollout_ref.model.exp_model_path)
            self.gpt_rm_client = GptRMClient(exp_model_path_str)
            self.exp_learner_wg = None
        elif self.skip_rm_scoring:
            print("[Init] skip_rm_scoring=True, no exp_learner_wg created")
            self.exp_learner_wg = None
        else:
            self.exp_learner_wg = all_wg["exp_learner"]
            self.exp_learner_wg.init_model()

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
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        iter_teacher_steps = self.config.trainer.get("iter_teacher_steps", -1)
        if iter_teacher_steps > 0 and self.use_reference_policy and hasattr(self, 'ref_policy_wg'):
            ref_step = (self.global_steps // iter_teacher_steps) * iter_teacher_steps
            ref_step_file = os.path.join(local_global_step_folder, "ref_policy_step.txt")
            with open(ref_step_file, "w") as f:
                f.write(str(ref_step))
            print(f"[Save] ref_policy corresponds to actor@step{ref_step}")

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        if getattr(self, "train_dataloader", None) is not None:
            torch.save(self.train_dataloader.state_dict(), dataloader_local_path)
        elif getattr(self, "train_dataloaders", None) is not None:
            torch.save([dl.state_dict() for dl in self.train_dataloaders], dataloader_local_path)

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
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # The fixed exp learner is initialized from exp_model_path and has no checkpoint state.
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # restore ref_policy when using iterative teacher
        iter_teacher_steps = self.config.trainer.get("iter_teacher_steps", -1)
        if iter_teacher_steps > 0 and self.use_reference_policy and hasattr(self, 'ref_policy_wg'):
            ref_step_file = os.path.join(global_step_folder, "ref_policy_step.txt")
            if os.path.exists(ref_step_file):
                with open(ref_step_file) as f:
                    ref_step = int(f.read().strip())
                if ref_step == 0:
                    print(f"[Resume] ref_policy_step=0, ref has never been iterated, stays as base model")
                else:
                    ref_actor_path = os.path.join(os.path.dirname(global_step_folder), f"global_step_{ref_step}", "actor")
                    if os.path.isdir(ref_actor_path):
                        print(f"[Resume] Loading ref_policy from actor@step{ref_step} ({ref_actor_path})")
                        self.ref_policy_wg.load_checkpoint(ref_actor_path)
                    else:
                        print(f"[Resume] WARNING: actor checkpoint for ref (step{ref_step}) not found at {ref_actor_path}, ref stays as base model")
            else:
                print(f"[Resume] WARNING: ref_policy_step.txt not found, ref stays as base model")

        # load dataloader
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            saved = torch.load(dataloader_local_path, weights_only=False)
            if getattr(self, "train_dataloader", None) is not None and isinstance(saved, dict):
                self.train_dataloader.load_state_dict(saved)
            elif getattr(self, "train_dataloaders", None) is not None and isinstance(saved, list):
                for dl, sd in zip(self.train_dataloaders, saved):
                    dl.load_state_dict(sd)
            print(f"Loaded dataloader state from {dataloader_local_path}")
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

        rm_prompt_version = self.config.trainer["rm_prompt_version"]

        _RUBRIC_BASED_PROMPT_TEMPLATE_V1 = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.

<prompt>
{instruction}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

First, analyze the response against each rubric item, discussing how well the response meets or fails each criterion. Then, provide your final score as an integer between 1 and 10, wrapped in <score> and </score> tags.
Example ending:
<score> your_integer_score_from_1_to_10 </score>

Your evaluation:"""

        _EXP_MODEL_CONTEXT_TEMPLATE_V1 = """Here is a previous attempt and evaluation for this problem:

[Previous Response]
{student_response}

[Evaluation by Scoring Model]
{scoring_output}

Now, given the above evaluation, please solve the problem again with improvements:
{prompt}"""

        _RUBRIC_BASED_PROMPT_TEMPLATE_V2 = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.

<prompt>
{instruction}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

First, analyze the response against each rubric item, discussing how well the response meets or fails each criterion. Then, provide your final score as an integer between 1 and 10, wrapped in <score> and </score> tags.

After the score, distill your analysis into transferable experiential knowledge which is general, high-level, widely applicable insights that would help improve future responses to similar tasks. Focus on reusable strategies and patterns rather than details specific to this particular response. Output this knowledge wrapped in <experience> and </experience> tags.

Example ending:
<score> your_integer_score_from_1_to_10 </score>

<experience>
some experiential knowledge...
</experience>

Your evaluation:"""

        _EXP_MODEL_CONTEXT_TEMPLATE_V2 = """Here is some experiential knowledge:

<experience>
{experience}
</experience>

Given the above experiential knowledge, solve the following problem:
{prompt}"""

        if self.config.trainer.stage == "rl":
            if rm_prompt_version != "v1":
                raise ValueError(
                    f"trainer.stage=rl requires rm_prompt_version=v1, got {rm_prompt_version}"
                )
            _RUBRIC_BASED_PROMPT_TEMPLATE = _RUBRIC_BASED_PROMPT_TEMPLATE_V1
            _EXP_MODEL_CONTEXT_TEMPLATE = _EXP_MODEL_CONTEXT_TEMPLATE_V1
        elif self.config.trainer.stage == "consolidate":
            if rm_prompt_version != "v2":
                raise ValueError(
                    "trainer.stage=consolidate requires rm_prompt_version=v2, "
                    f"got {rm_prompt_version}"
                )
            _RUBRIC_BASED_PROMPT_TEMPLATE = _RUBRIC_BASED_PROMPT_TEMPLATE_V2
            _EXP_MODEL_CONTEXT_TEMPLATE = _EXP_MODEL_CONTEXT_TEMPLATE_V2
        else:
            raise ValueError(f"Unknown trainer stage: {self.config.trainer.stage}")

        self._rubric_based_prompt_template = _RUBRIC_BASED_PROMPT_TEMPLATE
        self._exp_model_context_template = _EXP_MODEL_CONTEXT_TEMPLATE

        self._rm_templates = {
            "v2": (_RUBRIC_BASED_PROMPT_TEMPLATE_V2, _EXP_MODEL_CONTEXT_TEMPLATE_V2),
            "empty": (None, "{prompt}"),
        }
        if self.multidata_rm_prompt is not None:
            for ver in self.multidata_rm_prompt:
                if ver not in ("v2", "empty"):
                    raise ValueError(
                        f"[MultiData] multidata_rm_prompt only supports 'v2' and 'empty', got '{ver}'"
                    )


        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        start_epoch = (self.global_steps - 1) // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
        if start_epoch > 0:
            print(f"[Resume] Skipping {start_epoch} completed epoch(s), starting from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            if self.train_dataloaders is not None:
                dataloader_iter = zip(*self.train_dataloaders)
            else:
                dataloader_iter = self.train_dataloader
            for batch_dict in dataloader_iter:
                if isinstance(batch_dict, tuple):
                    batch_dict = self._concat_batch_dicts(batch_dict)

                if self.config.trainer.stage == "rl":
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
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    # pop those keys for generation
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "raw_prompt"]
                    if "tools_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("tools_kwargs")
                    if "interaction_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            rm_batch_size = batch.batch['responses'].shape[0]
                            rm_prompt_texts = []

                            for i in range(rm_batch_size):
                                msgs = batch.non_tensor_batch['raw_prompt'][i]
                                if isinstance(msgs, list):
                                    instruction = msgs[-1]['content']
                                else:
                                    instruction = str(msgs)

                                response_text = self.tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=True)
                                rubric_list_string = batch.non_tensor_batch['rubric_list_string'][i]

                                rm_prompt = _RUBRIC_BASED_PROMPT_TEMPLATE.format(
                                    instruction=instruction,
                                    response=response_text,
                                    rubric_list_string=rubric_list_string,
                                )
                                rm_prompt_texts.append(rm_prompt)

                            if self.use_gpt_rm:
                                rm_response_texts = self.gpt_rm_client.batch_call(rm_prompt_texts)
                            else:
                                rm_prompt_token_ids_list = []
                                for rm_prompt in rm_prompt_texts:
                                    rm_msgs = [{"role": "user", "content": rm_prompt}]
                                    rm_prompt_with_template = self.tokenizer.apply_chat_template(
                                        rm_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                    )
                                    rm_tokenized = self.tokenizer(
                                        rm_prompt_with_template, return_tensors="pt", add_special_tokens=False,
                                        padding=False, truncation=True, max_length=self.config.data["max_prompt_length"],
                                    )
                                    rm_prompt_token_ids_list.append(rm_tokenized["input_ids"][0].tolist())

                                rm_batch = self._build_dataproto_woresp(rm_prompt_token_ids_list)
                                rm_batch.meta_info = {
                                    "eos_token_id": self.tokenizer.eos_token_id,
                                    "pad_token_id": self.tokenizer.pad_token_id,
                                    "recompute_log_prob": False,
                                    "do_sample": True,
                                    "validate": True,
                                    "n": 1,
                                }
                                rm_batch_padded, rm_pad_size = pad_dataproto_to_divisor(rm_batch, self.exp_learner_wg.world_size)
                                rm_output_padded = self.exp_learner_wg.generate_sequences(rm_batch_padded)
                                rm_output = unpad_dataproto(rm_output_padded, rm_pad_size)
                                rm_response_texts = [self.tokenizer.decode(rm_output.batch["responses"][i], skip_special_tokens=True) for i in range(rm_batch_size)]

                            reward_scores = []
                            rm_valid_count = 0
                            rm_response_lengths = []
                            rm_truncated_count = 0
                            max_resp_len = self.config.data.max_response_length
                            for i in range(rm_batch_size):
                                rm_response_text = rm_response_texts[i]
                                if (self.global_steps == 1 or self.global_steps % 5 == 0) and i == 0:
                                    print(f'[Step {self.global_steps}] RM output:')
                                    print(rm_response_text)
                                # Tokenize with self.tokenizer for length stats
                                rm_resp_token_ids = self.tokenizer.encode(rm_response_text, add_special_tokens=False)
                                if len(rm_resp_token_ids) > max_resp_len:
                                    rm_truncated_count += 1
                                    rm_resp_token_ids = rm_resp_token_ids[:max_resp_len]
                                    rm_response_text = self.tokenizer.decode(rm_resp_token_ids, skip_special_tokens=True)
                                rm_response_lengths.append(len(rm_resp_token_ids))
                                score = 1.0
                                score_match = re.search(r'<score>(.*?)</score>', rm_response_text, re.DOTALL | re.IGNORECASE)
                                if score_match:
                                    score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_match.group(1).strip())
                                    if score_number_match:
                                        try:
                                            score = float(score_number_match.group(1))
                                            score = max(1.0, min(10.0, score))
                                            rm_valid_count += 1
                                        except ValueError:
                                            score = 1.0
                                reward_scores.append(score)

                            reward_scores_1d = torch.tensor(reward_scores, dtype=torch.float32)
                            reward_scale = self.config.trainer.get("reward_scale", 1.0)
                            if reward_scale != 1.0:
                                reward_scores_1d = reward_scores_1d / reward_scale
                            response_length = batch.batch['responses'].shape[1]
                            reward_tensor = torch.zeros(rm_batch_size, response_length, dtype=torch.float32)
                            response_mask = batch.batch['response_mask']
                            for i in range(rm_batch_size):
                                valid_positions = response_mask[i].nonzero(as_tuple=False)
                                if len(valid_positions) > 0:
                                    last_pos = valid_positions[-1].item()
                                    reward_tensor[i, last_pos] = reward_scores_1d[i]
                                else:
                                    reward_tensor[i, -1] = reward_scores_1d[i]
                            reward_extra_infos_dict = {}

                            metrics.update({
                                "actor/curr_reward": reward_scores_1d.mean().item() * reward_scale / 10.0,
                                "actor/rm_valid_score_ratio": rm_valid_count / rm_batch_size if rm_batch_size > 0 else 0.0,
                                "response_length/rm_output_avg_len": sum(rm_response_lengths) / len(rm_response_lengths) if rm_response_lengths else 0.0,
                                "response_length/rm_output_max_len": max(rm_response_lengths) if rm_response_lengths else 0.0,
                                "response_length/rm_output_truncation_rate": rm_truncated_count / rm_batch_size if rm_batch_size > 0 else 0.0,
                            })

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            if self.config.actor_rollout_ref.actor.get("save_logprob", False):
                                old_log_prob.batch["old_entropys"] = old_log_prob.batch.pop("entropys")
                            else:
                                old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            if self.use_critic:
                                with marked_timer("values", timing_raw, color="cyan"):
                                    values_output = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values_output)

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                config=self.config.algorithm,
                            )

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["stage_merge"] = False
                                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                batch.meta_info["global_steps"] = self.global_steps
                                if self.config.actor_rollout_ref.actor.get("save_logprob", False):
                                    batch.meta_info["save_logprob_dir"] = os.path.join(self.config.trainer.default_local_dir, "saved_prob", f"global_step_{self.global_steps}")
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                            if self.use_critic:
                                with marked_timer("update_critic", timing_raw, color="magenta"):
                                    critic_output = self.critic_wg.update_critic(batch)
                                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    metrics.update(critic_output_metrics)

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

                        # Iterative teacher update: copy actor weights to ref_policy
                        iter_teacher_steps = self.config.trainer.get("iter_teacher_steps", -1)
                        if iter_teacher_steps > 0 and self.global_steps > 0 and self.global_steps % iter_teacher_steps == 0:
                            with marked_timer("iter_teacher_update", timing_raw, color="cyan"):
                                print(f"[Step {self.global_steps}] Updating ref_policy with actor weights (iter_teacher_steps={iter_teacher_steps})")
                                tmp_path = os.path.join(self.config.trainer.default_local_dir, "_iter_teacher_tmp")
                                self.actor_rollout_wg.save_checkpoint(tmp_path, None, self.global_steps)
                                self.ref_policy_wg.load_checkpoint(tmp_path)
                                print(f"[Step {self.global_steps}] ref_policy updated")

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
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

                elif self.config.trainer.stage == "consolidate":
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
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    _prompt_rm_vers = batch.non_tensor_batch.get('rm_prompt_version', None)
                    # pop those keys for generation
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "raw_prompt"]
                    if "tools_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("tools_kwargs")
                    if "interaction_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        n_rollout = self.config.actor_rollout_ref.rollout.n

                        # 1) Generate student responses without context
                        with marked_timer("student_gen", timing_raw, color="cyan"):
                            student_gen_batch = gen_batch.select(deepcopy=True)
                            student_gen_batch.non_tensor_batch.pop("raw_prompt_ids", None)
                            student_gen_batch.non_tensor_batch.pop("raw_prompt", None)
                            student_output = self.actor_rollout_wg.generate_sequences(student_gen_batch)

                        # Per-sample rm_prompt_version for multi-dataset
                        n_samples = student_output.batch['responses'].shape[0]
                        if _prompt_rm_vers is not None:
                            sample_rm_vers = [str(_prompt_rm_vers[i // n_rollout]) for i in range(n_samples)]
                        else:
                            sample_rm_vers = None

                        # 2) Score each v2 student response with the rubric.
                        # Samples using the multidata "empty" setting skip RM scoring.
                        rm_prompt_texts = []
                        rm_sample_indices = []
                        for i in range(n_samples):
                            ver_i = sample_rm_vers[i] if sample_rm_vers is not None else rm_prompt_version
                            if ver_i == "empty":
                                continue
                            pi = i // n_rollout
                            msgs = gen_batch.non_tensor_batch['raw_prompt'][pi]
                            if isinstance(msgs, list):
                                instruction = msgs[-1]['content']
                            else:
                                instruction = str(msgs)
                            response_text = self.tokenizer.decode(student_output.batch['responses'][i], skip_special_tokens=True)
                            rubric_list_string = batch.non_tensor_batch['rubric_list_string'][pi]
                            rubric_tmpl = self._rm_templates[ver_i][0]
                            rm_prompt = rubric_tmpl.format(
                                instruction=instruction,
                                response=response_text,
                                rubric_list_string=rubric_list_string,
                            )
                            rm_prompt_texts.append(rm_prompt)
                            rm_sample_indices.append(i)

                        if rm_prompt_texts:
                            with marked_timer("exp_model_score", timing_raw, color="magenta"):
                                if self.use_gpt_rm:
                                    rm_response_texts_subset = self.gpt_rm_client.batch_call(rm_prompt_texts)
                                else:
                                    rm_prompt_token_ids_list = []
                                    for rm_prompt in rm_prompt_texts:
                                        rm_msgs = [{"role": "user", "content": rm_prompt}]
                                        rm_prompt_with_template = self.tokenizer.apply_chat_template(
                                            rm_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                        )
                                        rm_tokenized = self.tokenizer(
                                            rm_prompt_with_template, return_tensors="pt",
                                            add_special_tokens=False, padding=False, truncation=True,
                                            max_length=self.config.data["max_prompt_length"],
                                        )
                                        rm_prompt_token_ids_list.append(rm_tokenized["input_ids"][0].tolist())

                                    rm_batch = self._build_dataproto_woresp(rm_prompt_token_ids_list)
                                    rm_batch.meta_info = {
                                        "eos_token_id": self.tokenizer.eos_token_id,
                                        "pad_token_id": self.tokenizer.pad_token_id,
                                        "recompute_log_prob": False,
                                        "do_sample": True,
                                        "validate": True,
                                        "n": 1,
                                    }
                                    rm_batch_padded, rm_pad_size = pad_dataproto_to_divisor(rm_batch, self.exp_learner_wg.world_size)
                                    rm_output_padded = self.exp_learner_wg.generate_sequences(rm_batch_padded)
                                    rm_output = unpad_dataproto(rm_output_padded, rm_pad_size)
                                    rm_response_texts_subset = [
                                        self.tokenizer.decode(response, skip_special_tokens=True)
                                        for response in rm_output.batch["responses"]
                                    ]

                                # Expand subset results back to full array
                                rm_response_texts = [""] * n_samples
                                for idx, orig_i in enumerate(rm_sample_indices):
                                    rm_response_texts[orig_i] = rm_response_texts_subset[idx]
                        else:
                            rm_response_texts = [""] * n_samples

                        # 3) Parse RM scores for logging
                        exp_model_reward_scores = []
                        exp_rm_valid_count = 0
                        exp_rm_response_lengths = []
                        exp_rm_truncated_count = 0
                        max_resp_len = self.config.data.max_response_length
                        for i in range(n_samples):
                            ver_i = sample_rm_vers[i] if sample_rm_vers is not None else rm_prompt_version
                            if ver_i == "empty":
                                exp_model_reward_scores.append(0.0)
                                exp_rm_response_lengths.append(0)
                                continue
                            rm_response_text = rm_response_texts[i]
                            if (self.global_steps == 1 or self.global_steps % 5 == 0) and i == 0:
                                print(f'[Step {self.global_steps}] RM output:')
                                print(rm_response_text)
                            rm_resp_token_ids = self.tokenizer.encode(rm_response_text, add_special_tokens=False)
                            if len(rm_resp_token_ids) > max_resp_len:
                                exp_rm_truncated_count += 1
                                rm_resp_token_ids = rm_resp_token_ids[:max_resp_len]
                                rm_response_text = self.tokenizer.decode(rm_resp_token_ids, skip_special_tokens=True)
                                rm_response_texts[i] = rm_response_text
                            exp_rm_response_lengths.append(len(rm_resp_token_ids))
                            score = 1.0
                            score_match = re.search(r'<score>(.*?)</score>', rm_response_text, re.DOTALL | re.IGNORECASE)
                            if score_match:
                                score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_match.group(1).strip())
                                if score_number_match:
                                    try:
                                        score = float(score_number_match.group(1))
                                        score = max(1.0, min(10.0, score))
                                        exp_rm_valid_count += 1
                                    except ValueError:
                                        score = 1.0
                            exp_model_reward_scores.append(score)

                        # 4) Build per-sample experience from RM output
                        EXPERIENCES = []
                        for i in range(n_samples):
                            ver_i = sample_rm_vers[i] if sample_rm_vers is not None else rm_prompt_version
                            if ver_i == "empty":
                                EXPERIENCES.append("")
                            else:
                                scoring_output = rm_response_texts[i]
                                exp_match = re.search(r'<experience>(.*?)</experience>', scoring_output, re.DOTALL | re.IGNORECASE)
                                if exp_match:
                                    experience_text = exp_match.group(1).strip()
                                else:
                                    experience_text = ""
                                EXPERIENCES.append(experience_text)

                        if self.global_steps == 1:
                            print(
                                f"[ExpModel] Generated {n_samples} per-sample experiences "
                                f"(n_rollout={n_rollout}, rm_prompt_version={rm_prompt_version}, "
                                f"multi={sample_rm_vers is not None})"
                            )
                            print(f"[ExpModel] Sample 0 experience: {str(EXPERIENCES[0])[:300]}...")

                        with marked_timer("gen", timing_raw, color="red"):
                            exp_repeat = n_rollout
                            gen_batch_with_exp = gen_batch.select(deepcopy=True)
                            gen_batch_with_exp = gen_batch_with_exp.repeat(repeat_times=exp_repeat, interleave=True)
                            batch_with_exp = batch.select(deepcopy=True)
                            batch_with_exp = batch_with_exp.repeat(repeat_times=exp_repeat, interleave=True)

                            updated_gen_inputs = []
                            for i in range(len(gen_batch_with_exp)):
                                msgs = deepcopy(gen_batch_with_exp.non_tensor_batch['raw_prompt'][i])

                                content = msgs[-1]['content']
                                # Per-sample rm_prompt_version for multi-dataset
                                if _prompt_rm_vers is not None:
                                    pi_ctx = i // exp_repeat
                                    ver_i = str(_prompt_rm_vers[pi_ctx])
                                else:
                                    ver_i = rm_prompt_version
                                _, ctx_tmpl = self._rm_templates[ver_i]
                                if ver_i == "empty":
                                    updated_content = ctx_tmpl.format(prompt=content)
                                else:
                                    experience_text = EXPERIENCES[i]
                                    updated_content = ctx_tmpl.format(
                                        experience=experience_text,
                                        prompt=content,
                                    )
                                msgs[-1]['content'] = updated_content
                                tokenized = self.train_dataset.re_tokenize(msgs)
                                if (self.global_steps == 1 or self.global_steps % 5 == 0) and i == 0:
                                    print(f'[Step {self.global_steps}] Prompt with experience: ')
                                    print(self.tokenizer.decode(tokenized["input_ids"].tolist(), skip_special_tokens=True))
                                updated_gen_inputs.append(tokenized)

                            gen_batch_with_exp.batch["input_ids"] = torch.stack([inp["input_ids"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.batch["attention_mask"] = torch.stack([inp["attention_mask"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.batch["position_ids"] = torch.stack([inp["position_ids"] for inp in updated_gen_inputs])
                            gen_batch_with_exp.non_tensor_batch.pop("raw_prompt_ids", None)
                            gen_batch_with_exp.non_tensor_batch.pop("raw_prompt", None)

                            gen_batch_output = student_output

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=n_rollout, interleave=True)
                        batch = batch.union(gen_batch_output)
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        # Construct batch_with_exp: use gen_batch_with_exp's prompts + gen_batch_output's responses
                        # Repeat gen_batch_with_exp and batch_with_exp to match rollout.n (skip if use_exp_model, already pre-repeated)

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
                                reward_scores_1d = torch.tensor(exp_model_reward_scores, dtype=torch.float32)
                                # For logging, exclude "empty" samples that got score=0.0
                                scored_mask = reward_scores_1d > 0.0
                                scored_rewards = reward_scores_1d[scored_mask]
                                n_scored = scored_mask.sum().item()
                                metrics.update({
                                    "actor/curr_reward": scored_rewards.mean().item() / 10.0 if n_scored > 0 else 0.0,
                                    "actor/rm_valid_score_ratio": exp_rm_valid_count / n_scored if n_scored > 0 else 0.0,
                                    "response_length/rm_output_avg_len": sum(exp_rm_response_lengths) / len(exp_rm_response_lengths) if exp_rm_response_lengths else 0.0,
                                    "response_length/rm_output_max_len": max(exp_rm_response_lengths) if exp_rm_response_lengths else 0.0,
                                    "response_length/rm_output_truncation_rate": exp_rm_truncated_count / n_scored if n_scored > 0 else 0.0,
                                })
                        # get topk logits indices
                        if self.config.actor_rollout_ref.actor.kl_loss_type == "full" and self.config.actor_rollout_ref.actor.kl_topk > 0:
                            batch.meta_info["return_all_logits"] = True
                            batch_with_exp.meta_info["return_all_logits"] = True
                            with marked_timer("compute_topk_indices", timing_raw, color="purple"):
                                log_prob_proto = self.actor_rollout_wg.compute_log_prob(batch)
                                log_probs = log_prob_proto.batch["old_log_probs"]

                                actor_topk_indices = log_probs.long()
                                jsd_beta = self.config.actor_rollout_ref.actor.get("jsd_beta", -1)
                                if jsd_beta > 0:
                                    batch_with_exp.batch["first_kl_topk_indices"] = actor_topk_indices
                                    ref_log_prob_for_topk = self.ref_policy_wg.compute_ref_log_prob(batch_with_exp)
                                    batch_with_exp.batch["kl_topk_indices"] = ref_log_prob_for_topk.batch[
                                        "ref_log_prob"
                                    ].long()
                                    del ref_log_prob_for_topk
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
                                kl_loss_type = self.config.actor_rollout_ref.actor.kl_loss_type
                                kl_topk = self.config.actor_rollout_ref.actor.kl_topk
                                use_chunked_full_kl = (kl_loss_type == "full" and kl_topk <= 0)
                                if use_chunked_full_kl:
                                    batch_with_exp.meta_info["return_hidden_states"] = True
                                    batch_with_exp.meta_info["return_all_logits"] = False
                                else:
                                    batch_with_exp.meta_info["return_all_logits"] = kl_loss_type == "full"
                                assert self.use_reference_policy
                                if self.ref_policy_static_wg is not None and "use_fixed_teacher" in batch_with_exp.non_tensor_batch:
                                    fixed_mask = batch_with_exp.non_tensor_batch["use_fixed_teacher"].astype(int)
                                    segments = []
                                    i = 0
                                    while i < len(fixed_mask):
                                        val = fixed_mask[i]
                                        j = i
                                        while j < len(fixed_mask) and fixed_mask[j] == val:
                                            j += 1
                                        segments.append((i, j, bool(val)))
                                        i = j
                                    results = []
                                    for start, end, is_fixed in segments:
                                        seg_batch = batch_with_exp[start:end]
                                        if is_fixed:
                                            results.append(self.ref_policy_static_wg.compute_ref_log_prob(seg_batch))
                                        else:
                                            results.append(self.ref_policy_wg.compute_ref_log_prob(seg_batch))
                                    exp_log_prob = DataProto.concat(results) if len(results) > 1 else results[0]
                                else:
                                    exp_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch_with_exp)

                                if "entropys" in exp_log_prob.batch:
                                    exp_entropys = exp_log_prob.batch["entropys"]
                                    response_masks = batch_with_exp.batch["response_mask"]
                                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                    entropy_agg = agg_loss(loss_mat=exp_entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                    exp_prob_metrics = {"actor/exp_entropy": entropy_agg.detach().item()}
                                    metrics.update(exp_prob_metrics)
                                    exp_log_prob.batch.pop("entropys")
                                if use_chunked_full_kl:
                                    # Rename ref_log_prob -> ref_hidden_states (shape is bs, R, D)
                                    exp_log_prob.batch["ref_hidden_states"] = exp_log_prob.batch.pop("ref_log_prob")
                                else:
                                    exp_log_prob.batch["exp_log_probs"] = exp_log_prob.batch["ref_log_prob"]
                                    exp_log_prob.batch.pop("ref_log_prob")

                                if kl_loss_type == "full" and kl_topk > 0:
                                    exp_log_prob.batch["kl_topk_indices"] = batch_with_exp.batch["kl_topk_indices"]

                                batch = batch.union(exp_log_prob)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["stage_merge"] = True
                                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                batch.meta_info["global_steps"] = self.global_steps
                                if self.config.actor_rollout_ref.actor.get("save_logprob", False):
                                    batch.meta_info["save_logprob_dir"] = os.path.join(self.config.trainer.default_local_dir, "saved_prob", f"global_step_{self.global_steps}")
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

                        # Iterative teacher update: copy actor weights to ref_policy
                        iter_teacher_steps = self.config.trainer.get("iter_teacher_steps", -1)
                        if iter_teacher_steps > 0 and self.global_steps > 0 and self.global_steps % iter_teacher_steps == 0:
                            with marked_timer("iter_teacher_update", timing_raw, color="cyan"):
                                print(f"[Step {self.global_steps}] Updating ref_policy with actor weights (iter_teacher_steps={iter_teacher_steps})")
                                tmp_path = os.path.join(self.config.trainer.default_local_dir, "_iter_teacher_tmp")
                                self.actor_rollout_wg.save_checkpoint(tmp_path, None, self.global_steps)
                                self.ref_policy_wg.load_checkpoint(tmp_path)
                                print(f"[Step {self.global_steps}] ref_policy updated")

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
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
