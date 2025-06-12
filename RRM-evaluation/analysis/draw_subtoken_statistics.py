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

import argparse
from pathlib import Path

from analysis.visualization import draw_subtoken_statistics
from rewardbench.constants import SUBSET_MAPPING


def get_args():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("output_path", type=Path, help="Path to save the generated figure.")
    # optional arguments
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="oobabooga/llama-tokenizer",
        help="Pointer to the HuggingFace repository to source the tokenizer.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/reward-bench",
        help="Pointer to the HuggingFace repository that contains the benchmark dataset.",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[6, 12],
        help="Control the figure size when plotting.",
    )
    parser.add_argument(
        "--render_latex",
        action="store_true",
        help="If set, then it will render a LaTeX string instead of Markdown.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    draw_subtoken_statistics(
        category_subsets=SUBSET_MAPPING,
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path,
        figsize=args.figsize,
    )


if __name__ == "__main__":
    main()
