# Copyright 2023 AllenAI. All rights reserved.
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

# Draw the per token reward
# Note, requires pip install spacy-alignments

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import spacy_alignments as tokenizations

from analysis.visualization import draw_per_token_reward

DEFAULT_DIRNAME = "per-token-reward"


def get_args():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("text_hash", type=str, help="Path or pointer to the text hash to plot.")
    parser.add_argument("output_path", type=Path, help="Filepath to save the generated figure.")
    # optional arguments
    parser.add_argument(
        "--local",
        action="store_true",
        help="Find the file locally.",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[8, 8],
        help="Control the figure size when plotting.",
    )
    parser.add_argument(
        "--line_chart",
        action="store_true",
        help="Draw a line chart instead of a heatmap.",
    )
    parser.add_argument(
        "--do_not_align_tokens",
        action="store_true",
        help="If set, then tokens will not be aligned. May cause issues in the plot.",
    )
    args = parser.parse_args()
    return args


def align_tokens(reference_tokens: List[str], predicted_tokens: List[str], rewards: List[float]) -> List[float]:
    """Align tokens and recompute the reward

    reference_tokens (List[str]): the reference tokenization to base the alignment on.
    predicted_tokens (List[str]): the tokens from the reward pipeline.
    rewards (List[float]): the per-token reward.
    RETURNS (List[float]): the recomputed per-token reward.
    """
    a2b, _ = tokenizations.get_alignments(reference_tokens, predicted_tokens)
    rewards_list = []
    for aligned_idxs in a2b:
        rewards_list.append([rewards[idx] for idx in aligned_idxs])
    aligned_rewards = list(map(np.mean, rewards_list))
    return aligned_rewards


def main():
    args = get_args()
    # Read the results first
    input_dir = Path.cwd() / DEFAULT_DIRNAME / args.text_hash
    assert input_dir.exists(), f"Directory {input_dir} does not exist!"

    rewards = {}
    for file in input_dir.glob("*.json"):
        with open(file) as f:
            results = json.load(f)
            rewards[results["model"]] = results

    assert len(rewards) > 0, f"Directory {input_dir} is empty!"

    # Get reference alignment
    first_key = next(iter(rewards))  # should be the same all throughout
    text = rewards[first_key]["text"]
    whitespace_tokenizer = lambda x: x.split(" ")  # noqa
    reference_tokens = whitespace_tokenizer(text)

    if not args.do_not_align_tokens:
        for _, results in rewards.items():
            results["aligned_rewards"] = align_tokens(
                reference_tokens=reference_tokens,
                predicted_tokens=results["tokens"],
                rewards=results["rewards"],
            )

    reward_key = "rewards" if args.do_not_align_tokens else "aligned_rewards"
    draw_per_token_reward(
        tokens=reference_tokens,
        rewards=[reward[reward_key] for _, reward in rewards.items()],
        model_names=[model_name for model_name, _ in rewards.items()],
        output_path=args.output_path,
        figsize=args.figsize,
        line_chart=args.line_chart,
    )


if __name__ == "__main__":
    main()
