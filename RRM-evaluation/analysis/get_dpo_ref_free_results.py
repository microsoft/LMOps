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

# Script for getting DPO ref free results

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from analysis.utils import load_results
from rewardbench.constants import (
    EXAMPLE_COUNTS,
    SUBSET_MAPPING,
    SUBSET_NAME_TO_PAPER_READY,
)

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
        default=None,
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--render_latex",
        action="store_true",
        help="If set, then it will render a LaTeX string instead of Markdown.",
    )
    parser.add_argument(
        "--ignore_columns",
        type=lambda x: x.split(",") if x is not None else None,
        default=None,
        help="Comma-separated column names to exclude from the report.",
    )
    args = parser.parse_args()
    return args


def get_average_over_rewardbench(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Get average over a strict subset of reward models"""
    new_df = df.copy()
    for subset, sub_subsets in SUBSET_MAPPING.items():
        subset_cols = [col for col in new_df.columns if col in sub_subsets]
        sub_data = new_df[subset_cols].values  # take the relevant column values
        sub_counts = [EXAMPLE_COUNTS[s] for s in sub_subsets]  # take the example counts
        new_df[subset] = np.average(sub_data, axis=1, weights=sub_counts)

    data_cols = list(SUBSET_MAPPING.keys())
    keep_columns = ["model"] + ["model_type"] + data_cols
    new_df = new_df[keep_columns]

    # add total average
    new_df["average"] = np.nanmean(new_df[data_cols].values, axis=1)

    # make average third column
    keep_columns = ["model", "model_type", "average"] + data_cols
    new_df = new_df[keep_columns]
    return new_df


def main():
    args = get_args()

    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN not found!")

    print(f"Downloading repository snapshots into '{LOCAL_DIR}' directory")
    # Load the remote repository using the HF API
    hf_evals_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR) / "rewardbench",
        repo_id=args.hf_evals_repo,
        use_auth_token=api_token,
        ignore_patterns=[
            "eval-set/*",
        ],
        tqdm_class=None,
        etag_timeout=30,
        repo_type="dataset",
    )
    hf_evals_df = load_results(
        hf_evals_repo, subdir="eval-set/", ignore_columns=args.ignore_columns, remove_ref_free=False
    )

    # select only the rows where model_type == DPO
    df_dpo = hf_evals_df[hf_evals_df["model_type"] == "DPO"]

    # select only the rows where model_type == DPO Ref. Free
    df_dpo_ref_free = hf_evals_df[hf_evals_df["model_type"] == "DPO Ref. Free"]

    # if model is the same for any row of ref free, take the first row (its the default method)
    df_dpo_ref_free = df_dpo_ref_free.drop_duplicates(subset=["model"], keep="first")

    # drop rows from df_dpo if df_dpo_ref_free doesn't have that model
    df_dpo = df_dpo[df_dpo["model"].isin(df_dpo_ref_free["model"])]

    def _multiply_numbered_cols_by(n, df, ignore: List[str] = []):
        numbered_cols = df.select_dtypes("number").columns
        df[numbered_cols] *= n
        return df

    dpo_scaled = _multiply_numbered_cols_by(100, get_average_over_rewardbench(df_dpo))
    ref_free_scaled = _multiply_numbered_cols_by(100, get_average_over_rewardbench(df_dpo_ref_free))

    # order dpo_scaled and ref_free scaled by the models
    dpo_scaled = dpo_scaled.sort_values(by="model")
    ref_free_scaled = ref_free_scaled.sort_values(by="model")

    # create copy of the column "average" from ref_free_scaled
    ref_free_avg = dpo_scaled["average"].copy()
    new_avg = ref_free_scaled["average"].copy()

    # for every model, update ref_free_scaled to be the difference with dpo_scaled
    for model in dpo_scaled["model"]:
        # iterate over columns and update the values
        for col in ref_free_scaled.columns:
            if col != "model" and col != "model_type":
                ref_free_scaled.loc[ref_free_scaled["model"] == model, col] = (
                    ref_free_scaled.loc[ref_free_scaled["model"] == model, col].values[0]
                    - dpo_scaled.loc[dpo_scaled["model"] == model, col].values[0]
                )

    # rename column "average" to "delta"
    ref_free_scaled = ref_free_scaled.rename(columns={"average": "Delta"})

    # add ref_free_avg back as "Score"
    ref_free_scaled["Orig. Score"] = ref_free_avg

    # move Score to be the 3rd column
    cols = list(ref_free_scaled.columns)
    cols.insert(2, cols.pop(cols.index("Orig. Score")))
    ref_free_scaled = ref_free_scaled.loc[:, cols]

    # add column New Score after Orig. Score from new_avg
    ref_free_scaled["New Score"] = new_avg
    # move New Score to be the 4th column
    cols = list(ref_free_scaled.columns)
    cols.insert(3, cols.pop(cols.index("New Score")))
    ref_free_scaled = ref_free_scaled.loc[:, cols]

    # sort by Score (biggest at top)
    ref_free_scaled = ref_free_scaled.sort_values(by="Orig. Score", ascending=False)

    # remove model_type column
    ref_free_scaled = ref_free_scaled.drop(columns=["model_type"])

    df = ref_free_scaled.round(1)
    # df.insert(0, "", range(1, 1 + len(df)))

    df = df.rename(columns=SUBSET_NAME_TO_PAPER_READY)

    if args.render_latex:
        # Define a function to calculate color based on value
        def color_for_value(value):
            # Example: Map value to a shade of red, assuming Delta ranges from -1 to 1
            # Adjust the color scale according to your specific needs
            if np.isnan(value):
                return "\\cellcolor{gray!20}"  # Gray color for NaN values
            else:
                intensity = np.abs(value)  # Scale the value to [0, 100]
                if value > 0:
                    return f"\\cellcolor{{blue!{intensity:.0f}}}"  # Green for positive values
                else:
                    return f"\\cellcolor{{red!{intensity:.0f}}}"  # Red for negative values

        # Apply color formatting to the Delta column
        def _apply_delta_color(row, key="Delta"):
            delta_val = row[key]
            colored_delta = color_for_value(delta_val) + f" {delta_val:.1f}"
            return colored_delta

        # Assuming 'Delta' is a column in your dataframe
        df["Delta"] = df.apply(_apply_delta_color, axis=1)
        # apply delta color to Chat Chat Hard Safety and Reasoning too
        df["Chat"] = df.apply(_apply_delta_color, args=("Chat",), axis=1)
        df["Chat Hard"] = df.apply(_apply_delta_color, args=("Chat Hard",), axis=1)
        df["Safety"] = df.apply(_apply_delta_color, args=("Safety",), axis=1)
        df["Reasoning"] = df.apply(_apply_delta_color, args=("Reasoning",), axis=1)

        # Prettify: we're using openmojis instead of a model_type column
        def _prettify_model_name(row):
            orig_name = row["model"]

            latex_name = (
                f"\href{{https://huggingface.co/{orig_name}}}" + f"{{{orig_name}}}".replace("_", "\_")  # noqa  # noqa
                if orig_name != "random"
                else f"{orig_name}"
            )

            return latex_name

        reward_model_names = df.apply(lambda x: _prettify_model_name(x), axis=1).to_list()
        df.insert(0, "Reward Model", reward_model_names)
        df = df.drop(
            columns=[
                "model",
            ]
        )
        render_string = df.to_latex(index=False, float_format="%.1f").replace("NaN", "-")

    else:
        render_string = df.to_markdown(index=False, tablefmt="github")
        render_string = render_string.replace("NaN", "")
        render_string = render_string.replace("nan", "")

    print(render_string)

    if args.output_dir:
        print(f"Saving results to '{args.output_dir}/dpo_ref_free.csv'")
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        df.to_csv(args.output_dir / "dpo_ref_free.csv", index=False)


if __name__ == "__main__":
    main()
