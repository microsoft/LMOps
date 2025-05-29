# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   value_baseline: str = "optimal"):
    """
    Compute advantage for OPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    if value_baseline == "optimal":
        # use the optimal reward baseline for GRPO
        response_max_length = token_level_rewards.shape[-1]
        response_length = eos_mask.sum(dim=-1)
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2len = defaultdict(list)
        id2bsl = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
                id2len[index[i]].append(response_length[i])
            
            for idx in id2score:
                len_tensor = torch.tensor(id2len[idx])
                score_tensor = torch.tensor(id2score[idx])
                weighted_score = (len_tensor*score_tensor).sum() / len_tensor.sum()
                id2bsl[idx] = weighted_score

            for i in range(bsz):
                scores[i] = scores[i] - id2bsl[index[i]]
            scores = scores.unsqueeze(-1).tile([1, response_max_length]) * eos_mask
    elif value_baseline == "optimal_batch":
        # use the optimal reward baseline for Reinforce++, but exclude the KL reward
        response_max_length = token_level_rewards.shape[-1]
        response_length = eos_mask.sum(dim=-1)
        scores = token_level_rewards.sum(dim=-1)

        len_tensor = torch.tensor(response_length)
        score_tensor = torch.tensor(scores)
        bsl_score = (len_tensor*score_tensor).sum() / len_tensor.sum()
        scores = scores - bsl_score
        scores = scores.unsqueeze(-1).tile([1, response_max_length]) * eos_mask
    elif value_baseline == "grpo":
        # GRPO normalization
        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    elif value_baseline == "reinforce_plus_plus":
        # Reinforce++ normalization
        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        mean_score = torch.mean(scores)
        std_score = torch.std(scores)
        scores = (scores - mean_score) / (std_score + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    else:
        raise ValueError(f"unknown value baseline: {value_baseline}")

    return scores, scores
