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

# Script for converting RewardBench best of n (BoN) results into the AlpacaEval format

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

LOCAL_DIR = "hf_snapshot_evals"


def get_args():
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="allenai/reward-bench-results",
        help="HuggingFace repository containing the evaluation results.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="outputs/",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--generation_model",  # zephyr-7b or tulu-13b
        required=True,
        nargs=1,
        choices=["zephyr-7b", "tulu-13b"],
        help="The generation model used for the evaluation.",
    )
    parser.add_argument(
        "--reward_model",
        required=True,
        type=str,
        help="The reward model used for the evaluation.",
    )
    parser.add_argument(
        "--best_of",
        type=int,
        default=16,
        help="The number of responses to consider (from first index).",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Download the evaluation results
    # base_dir = "https://huggingface.co/datasets/ai2-adapt-dev/herm-debug/raw/main/best-of-n/alpaca_eval/"
    # d_file = base_dir + f"{args.generation_model[0]}" + f"/{args.reward_model}.json"
    # load dataset directly doesn't work with our schema for some reason
    # eval_data = load_dataset("json", data_files=d_file, split="train")

    hub_file = "best-of-n/alpaca_eval/" + f"{args.generation_model[0]}" + f"/{args.reward_model}.json"
    f = hf_hub_download(args.hf_evals_repo, hub_file, repo_type="dataset")
    eval_data = load_dataset("json", data_files=f, split="train")

    def split_dict_lists(input_dict, chunk_size=16):
        # List to hold the resulting dictionaries
        result = []

        # Iterate over each key-value pair in the input dictionary
        for key, value in input_dict.items():
            # Split the list into chunks of size 16
            for i in range(0, len(value), chunk_size):
                chunk = value[i : i + chunk_size]
                # Create a new dictionary for each chunk and add it to the result list
                result.append({key: chunk})

        return result

    # rename column prompt to 'instruction'
    eval_data = eval_data.rename_columns({"prompt": "instruction"})

    # add empty column input
    input_col = [""] * len(eval_data)
    eval_data = eval_data.add_column("input", input_col)
    # rename text to output
    eval_data = eval_data.rename_columns({"text": "output"})
    # rename model to generator
    eval_data = eval_data.rename_columns({"model": "generator"})

    # save locally to json for sending to AlpacaEval
    # create dir if needed
    out_dir = os.path.dirname(f"results/AlpacaEval/{args.generation_model[0]}-{args.reward_model}.json")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    eval_data.to_json(out_dir)


if __name__ == "__main__":
    main()
