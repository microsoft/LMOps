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

# Script to draw the distribution of model counts in a histogram

import argparse
from pathlib import Path

from analysis.visualization import draw_model_source_histogram, print_model_statistics


def get_args():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("output_path", type=Path, help="Filepath to save the generated figure.")
    # optional arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/reward-bench",
        help="The HuggingFace dataset name to source the eval dataset.",
    )
    parser.add_argument(
        "--keys",
        type=lambda x: x.split(","),
        default="chosen_model,rejected_model",
        help="Comma-separated columns to include in the histogram.",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 8],
        help="Control the figure size when plotting.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the values based on the total number of completions.",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Set the y-axis to a logarithmic scale.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only plot the top-n models in the histogram.",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    draw_model_source_histogram(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        keys=args.keys,
        figsize=args.figsize,
        normalize=args.normalize,
        log_scale=args.log_scale,
        top_n=args.top_n,
    )
    print_model_statistics()


if __name__ == "__main__":
    main()
