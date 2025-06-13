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

# Script for getting per model distributions across reward scores

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from analysis.utils import load_scores
from analysis.visualization import AI2_COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)

LOCAL_DIR = "./hf_snapshot_evals/"


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
        default="plots/",
        help="Directory to save the results.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN not found!")

    print(f"Downloading repository snapshots into '{LOCAL_DIR}' directory")
    # Load the remote repository using the HF API
    hf_evals_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR),
        repo_id=args.hf_evals_repo,
        ignore_patterns=["pref-sets/*", "eval-set/*"],
        use_auth_token=api_token,
        tqdm_class=None,
        repo_type="dataset",
    )
    hf_evals_df = load_scores(hf_evals_repo, subdir="eval-set-scores/")
    generate_whisker_plot(
        hf_evals_df,
        args.output_dir,
        model_type="Seq. Classifier",
        ncol=3,
        height=12,
        width=12,
        name="score-dist-seq-core",
    )
    generate_whisker_plot(
        hf_evals_df,
        args.output_dir,
        model_type="DPO",
        ncol=3,
        height=16,
        width=12,
        name="score-dist-dpo-core",
    )
    hf_prefs_df = load_scores(hf_evals_repo, subdir="pref-sets-scores/")
    generate_whisker_plot(
        hf_prefs_df,
        args.output_dir,
        model_type="Seq. Classifier",
        ncol=3,
        height=9,
        width=12,
        name="score-dist-seq-pref",
    )
    generate_whisker_plot(
        hf_prefs_df,
        args.output_dir,
        model_type="DPO",
        ncol=3,
        height=16,
        width=12,
        name="score-dist-dpo-pref",
    )


def generate_whisker_plot(df, output_path, model_type="Seq. Classifier", ncol=None, name=None, height=10, width=18):
    # select only the correct model type
    df = df[df["model_type"] == model_type]

    # get num_models
    models = df["model"].unique()
    n_models = len(models)

    # Calculate the number of rows and columns for the subplot grid
    if ncol is not None:
        ncols = ncol
        nrows = int(n_models / ncols) + (n_models % ncols > 0)
    else:
        nrows = int(n_models**0.5)
        ncols = int(n_models / nrows) + (n_models % nrows > 0)

    # Create a single figure and multiple subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))
    axs = axs.flatten()  # Flatten the array to iterate easily if it's 2D

    # Generate plots for each subset
    for i, model in enumerate(models):
        print(model)
        # if subset in ["donotanswer", "hep-cpp"]:
        #     import ipdb; ipdb.set_trace()
        # Filter data for the current subset
        subset_data = df[df["model"] == model]

        # take data from scores_chosen and scores_rejected and put into one scores array
        data_chosen = subset_data["scores_chosen"].values.tolist()
        data_rejected = subset_data["scores_rejected"].values.tolist()
        # flatten data if list of lists
        if isinstance(data_chosen[0], list):
            data_chosen = [item for sublist in data_chosen for item in sublist]
            data_rejected = [item for sublist in data_rejected for item in sublist]

        # print(len(data))

        # for ax[i] draw a histogram of the data
        axs[i].hist([data_chosen, data_rejected], bins=20, color=[AI2_COLORS["blue"], AI2_COLORS["orange"]], alpha=0.7)

        # ax title is model name (after /)
        axs[i].set_title(model.split("/")[-1])

    # Adjusting spines and setting ticks visibility
    for ax_idx, ax in enumerate(axs):
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Determine if the subplot is on the bottom row or the leftmost column
        # is_bottom = (ax_idx // ncols) == (nrows - 1) or nrows == 1
        is_left = (ax_idx % ncols) == 0

        # # Only show x-axis labels for bottom row subplots
        # ax.tick_params(axis="x", labelbottom=is_bottom)

        # Only show y-axis labels for leftmost column subplots
        ax.tick_params(axis="y", labelleft=is_left)

    # global y axis label
    fig.text(0.015, 0.5, "Density", va="center", rotation="vertical")

    # global x axis label
    fig.text(0.5, 0.015, "Reward Model Score", ha="center")

    bbox_anchor_y = -0.050 if name == "score-dist-seq-pref" else -0.040
    # global legend
    fig.legend(
        ["Chosen", "Rejected"],
        loc="lower center",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.5, bbox_anchor_y),
    )

    # Adjust layout and aesthetics
    # plt.suptitle("Per subset accuracy distribution", fontsize=16)
    plt.tight_layout(rect=[0.02, 0.01, 1, 1])  # Adjust layout to make room for the title
    plt.grid(False)

    # Handle any empty subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axs[j])

    # Show and/or save the plot
    if output_path:
        print(f"Saving figure to {output_path}")
        # if output path doesn't exist, make it
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / (name + ".pdf"), transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
