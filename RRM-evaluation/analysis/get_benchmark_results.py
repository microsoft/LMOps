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

# Script for getting reward model benchmark results

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
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to filter the results.",
    )
    parser.add_argument(
        "--print_all_results",
        action="store_true",
        default=False,
        help="If set, then it will render the full results.",
    )
    parser.add_argument(
        "--ignore_closed_models",
        action="store_true",
        default=False,
        help="If set, then it will ignore the closed models.",
    )
    args = parser.parse_args()
    return args


CLOSED_MODEL_LIST = [
    "Cohere May 2024",
    "Cohere March 2024",
]


def get_average_over_rewardbench(
    df: pd.DataFrame,
    df_prefs: pd.DataFrame,
    model_type: str = None,
) -> pd.DataFrame:
    """Get average over a strict subset of reward models"""
    new_df = df.copy()
    for subset, sub_subsets in SUBSET_MAPPING.items():
        subset_cols = [col for col in new_df.columns if col in sub_subsets]
        sub_data = new_df[subset_cols].values  # take the relevant column values
        sub_counts = [EXAMPLE_COUNTS[s] for s in subset_cols]  # take the example counts
        new_df[subset] = np.average(sub_data, axis=1, weights=sub_counts)

    data_cols = list(SUBSET_MAPPING.keys())
    keep_columns = ["model"] + ["model_type"] + data_cols
    new_df = new_df[keep_columns]

    # selected average from pref_sets
    pref_columns = ["anthropic_helpful", "anthropic_hhh", "shp", "summarize"]
    pref_data = df_prefs[pref_columns].values

    # add column test sets knowing the rows are not identical, take superset
    df_prefs["Prior Sets (0.5 weight)"] = np.nanmean(pref_data, axis=1)
    # add column Test Sets empty to new_df
    new_df["Prior Sets (0.5 weight)"] = np.nan
    # per row in new_df if model is in dataframe_prefs, add the value to new_df["Prior Sets"]
    values = []
    for i, row in new_df.iterrows():
        model = row["model"]
        if model in df_prefs["model"].values:
            values.append(df_prefs[df_prefs["model"] == model]["Prior Sets (0.5 weight)"].values[0])
            # new_df.at[i, "Prior Sets"] = dataframe_prefs[dataframe_prefs["model"] == model]["Prior Sets"].values[0]
        else:
            values.append(np.nan)

    new_df["Prior Sets (0.5 weight)"] = values

    # add total average
    data_cols += ["Prior Sets (0.5 weight)"]
    final_data = new_df[data_cols].values
    masked_data = np.ma.masked_array(final_data, np.isnan(final_data))
    weights = [2, 2, 2, 2, 1]
    average = np.ma.average(masked_data, axis=1, weights=weights)
    new_df["average"] = average.filled(np.nan)

    # make average third column
    keep_columns = ["model", "model_type", "average"] + data_cols
    new_df = new_df[keep_columns]

    # filter df from model_type in "Model Type"
    if model_type:
        new_df = new_df[new_df["model_type"] == model_type]
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
        ignore_patterns=["eval-set-scores/*", "pref-sets-scores/*"],
        tqdm_class=None,
        etag_timeout=30,
        repo_type="dataset",
    )
    hf_evals_df = load_results(hf_evals_repo, subdir="eval-set/", ignore_columns=args.ignore_columns)
    hf_prefs_df = load_results(hf_evals_repo, subdir="pref-sets/", ignore_columns=args.ignore_columns)

    # remove rows with closed models
    if args.ignore_closed_models:
        hf_evals_df = hf_evals_df[~hf_evals_df["model"].isin(CLOSED_MODEL_LIST)]
        hf_prefs_df = hf_prefs_df[~hf_prefs_df["model"].isin(CLOSED_MODEL_LIST)]

    def _multiply_numbered_cols_by(n, df, ignore: List[str] = []):
        numbered_cols = df.select_dtypes("number").columns
        df[numbered_cols] *= n
        return df

    all_results = {
        "RewardBench - Overview": _multiply_numbered_cols_by(
            100, get_average_over_rewardbench(hf_evals_df, hf_prefs_df, args.model_type)
        )
    }
    if args.print_all_results:
        all_results["RewardBench - Detailed"] = _multiply_numbered_cols_by(100, hf_evals_df)
        all_results["Pref Sets - Overview"] = _multiply_numbered_cols_by(100, hf_prefs_df)

        for category, subsets in SUBSET_MAPPING.items():
            df_per_category = hf_evals_df[subsets]
            df_per_category.insert(0, "model", hf_evals_df["model"].to_list())
            df_per_category.insert(1, "model_type", hf_evals_df["model_type"].to_list())

            wt_average = []
            for _, row in hf_evals_df[subsets].iterrows():
                scores = [row[s] for s in subsets]
                weights = [EXAMPLE_COUNTS.get(s) for s in subsets]
                wt_average.append(np.average(scores, weights=weights))

            df_per_category.insert(2, "average", wt_average)
            all_results[category] = df_per_category

    for name, df in all_results.items():
        # df.insert(0, "", range(1, 1 + len(df)))
        print(f"==================== {name} ====================")
        optional_header = """
        Reward Model & \thead{Avg} & \thead{Chat} & \thead{Chat\\Hard} & \thead{Safety} & \thead{Reason} & \thead{Prior\\Sets} \\
        """  # noqa
        print(optional_header)
        print("\n")

        df = df.sort_values(by="average", ascending=False).round(1)
        df = df.rename(columns=SUBSET_NAME_TO_PAPER_READY)

        if args.render_latex:
            # Prettify: we're using openmojis instead of a model_type column
            def _prettify_model_name(row):
                model_type = row["model_type"]
                orig_name = row["model"]
                openmoji_map = {
                    "Seq. Classifier": "\sequenceclf",  # noqa
                    "Custom Classifier": "\customclf",  # noqa
                    "DPO": "\dpo",  # noqa
                    "Generative": "\generative",  # noqa
                    "Generative RM": "\generative",  # noqa
                    "generative": "\generative",  # noqa
                }
                emoji = openmoji_map[model_type] if model_type in openmoji_map else "\\random"

                if "Cohere" in orig_name:
                    hf_name = "Cohere"
                elif "openai" in orig_name:
                    hf_name = "openai"
                elif "Anthropic" in orig_name:
                    hf_name = "Anthropic"
                else:
                    hf_name = orig_name

                # shorten long names
                if len(orig_name) > 50:
                    orig_name = orig_name[:48] + "..."

                latex_name = (
                    f"\href{{https://huggingface.co/{hf_name}}}"  # noqa
                    + f"{{{emoji} {orig_name}}}".replace("_", "\_")  # noqa
                    if orig_name != "random"
                    else f"{emoji} {orig_name}"
                )

                return latex_name

            reward_model_names = df.apply(lambda x: _prettify_model_name(x), axis=1).to_list()
            df.insert(0, "Reward Model", reward_model_names)
            df = df.drop(columns=["model", "model_type"]).rename(columns={"average": "Score"})
            if "Pref Sets" in name:
                df = df.drop(columns=["Prior Sets (0.5 weight)"])
            # Rotate column names using custom LaTeX command \rot
            df = df.rename(columns={col: "\\rot{" + col + "}" for col in df.columns})
            render_string = df.to_latex(index=False, float_format="%.1f").replace("NaN", "-")
        else:
            render_string = df.to_markdown(index=False, tablefmt="github")
        render_string = render_string.replace("NaN", "")
        render_string = render_string.replace("nan", "")
        print(name)
        print(render_string)

        if args.output_dir:
            print(f"Saving results to '{args.output_dir}/{name}.csv'")
            Path(args.output_dir).mkdir(exist_ok=True, parents=True)
            df.to_csv(args.output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
