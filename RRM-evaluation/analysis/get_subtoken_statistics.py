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
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
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
        "--split",
        type=str,
        default="filtered",
        help="Dataset split to use for obtaining the subtoken statistics.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--render_latex",
        action="store_true",
        help="If set, then it will render a LaTeX string instead of Markdown.",
    )
    args = parser.parse_args()
    return args


def get_dataset_tokens_per_subset(
    tokenizer_name: str,
    dataset_name: str,
    split: str,
) -> Dict[str, Dataset]:
    """Get subtokens from a dataset

    Expects that the dataset contains a 'prompt', 'chosen' and 'rejected'
    columns. It will then assign the tokenized list in the 'prompt_tokens',
    'chosen_tokens', and 'rejected_tokens', respectively.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset(
        dataset_name,
        download_mode="force_redownload",
        split=split,
        ignore_verifications=True,
    )

    subset_names = set(dataset["subset"])
    subsets = {s: dataset.filter(lambda x: x["subset"] == s) for s in subset_names}

    # Tokenize the text/s: some tokenizers like oobabooga adds a '1' padding
    # when calling the tokenizer() function directly---that's why we're
    # tokenizing it first to str, then calling convert_tokens_to_ids()
    def _tokenize(example):
        example["prompt_tokens"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example["prompt"]))
        example["chosen_tokens"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example["chosen"]))
        example["rejected_tokens"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example["rejected"]))
        return example

    return {s: d.map(_tokenize) for s, d in subsets.items()}


def main():
    args = get_args()
    subsets = get_dataset_tokens_per_subset(
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        split=args.split,
    )

    # We will always include the prompt when computing the token lengths for the
    # chosen and rejected responses
    def _get_statistics(dataset: Dataset) -> Dict[str, Any]:
        keys = ("chosen_lens", "rejected_lens", "chosen_unique_lens", "rejected_unique_lens")
        stats = {k: [] for k in keys}
        for eg in dataset:
            prompt_tokens = eg.get("prompt_tokens")
            chosen_tokens = eg.get("chosen_tokens")
            rejected_tokens = eg.get("rejected_tokens")

            stats["chosen_lens"].append(len(prompt_tokens) + len(chosen_tokens))
            stats["rejected_lens"].append(len(prompt_tokens) + len(rejected_tokens))
            # We compute the uniqueness across the whole instruction, NOT individually
            stats["chosen_unique_lens"].append(len(set(prompt_tokens + chosen_tokens)))
            stats["rejected_unique_lens"].append(len(set(prompt_tokens + rejected_tokens)))

        return stats

    subtoken_statistics = {name: _get_statistics(subset) for name, subset in subsets.items()}

    # Create report table
    df = pd.DataFrame(
        [
            {
                "subset": name,
                "Chosen Mean Tokens": np.mean(stats["chosen_lens"]),
                "Rejected Mean Tokens": np.mean(stats["rejected_lens"]),
                "Chosen Max Tokens": np.max(stats["chosen_lens"]),
                "Rejected Max Tokens": np.max(stats["rejected_lens"]),
                "Chosen Min Tokens": np.min(stats["chosen_lens"]),
                "Rejected Min Tokens": np.min(stats["rejected_lens"]),
                "Chosen Mean Unique Tokens": np.mean(stats["chosen_unique_lens"]),
                "Rejected Mean Unique Tokens": np.mean(stats["rejected_unique_lens"]),
                "Chosen Max Unique Tokens": np.max(stats["chosen_unique_lens"]),
                "Rejected Max Unique Tokens": np.max(stats["rejected_unique_lens"]),
                "Chosen Min Unique Tokens": np.min(stats["chosen_unique_lens"]),
                "Rejected Min Unique Tokens": np.min(stats["rejected_unique_lens"]),
            }
            for name, stats in subtoken_statistics.items()
        ]
    )

    # sort by subset
    df = df.sort_values(by="subset")

    render_string = (
        df.round(4).astype(str).to_latex(index=False)
        if args.render_latex
        else df.to_markdown(index=False, tablefmt="github")
    )
    render_string = render_string.replace("NaN", "")
    render_string = render_string.replace("nan", "")
    print(render_string)

    if args.output_dir:
        print(f"Saving results to '{args.output_dir}' directory")
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        df.to_csv(args.output_dir / "subtoken_statistics.csv", index=False)


if __name__ == "__main__":
    main()
