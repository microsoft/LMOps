# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import numpy as np

from verl.workers.reward_manager import register
from verl.workers.reward_manager import NaiveRewardManager

# from math_verify import parse, verify


def math_verify_reward_function(data_source, solution_str, ground_truth):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # We always take the final solution
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0

def math_verify_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, (str, float, int)):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, list) and isinstance(ground_truth[0], np.ndarray):
        ground_truth = ground_truth[0].tolist()
    score = math_verify_reward_function(data_source, solution_str, ground_truth)
    return float(score)


@register("hf_math_verify")
class HfMathVerifyRewardManager(NaiveRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the HfMathVerifyRewardManager instance.
        """
        super().__init__(tokenizer, num_examine, math_verify_compute_score, reward_fn_key)
