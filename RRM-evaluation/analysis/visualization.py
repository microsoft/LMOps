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

# Module for visualizing datasets and post-hoc analyses.

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from rewardbench.constants import SUBSET_NAME_TO_PAPER_READY

# From varnish: https://varnish.allenai.org/components/colors
AI2_COLORS = {
    "blue": "#265ed4",
    "light_blue": "#80bdff",
    "orange": "#dd6502",
    "light_orange": "#ffd45d",
    "red": "#932222",
    "light_red": "#ff9f9e",
    "aqua": "#054976",
    "light_aqua": "#b5f0ff",
    "teal": "#078e9e",
    "magenta": "#65295d",
    "purple": "#5c50a4",
    "green": "#005340",
}

# matplotlib params: use plt.rcParams.update(PLOT_PARAMS)
FONT_SIZES = {"small": 18, "medium": 21, "large": 24}


def _get_font() -> Optional[str]:
    system_fonts = matplotlib.font_manager.findSystemFonts()
    available_fonts = []
    try:
        for font in system_fonts:
            available_fonts.append(matplotlib.font_manager.get_font(font))
    except Exception:
        pass  # do nothing, we just want to get the fonts that work.
    if "Times New Roman" in available_fonts:
        return "Times New Roman"
    else:
        print("Font 'Times New Roman' not found, trying 'STIX'")
        if "STIX" in available_fonts:
            return "STIX"
        else:
            print("Font 'STIX' not found. To install, see: https://www.stixfonts.org/")
            print("Will use default fonts")
            return None


PLOT_PARAMS = {
    "font.family": "Times New Roman",
    "font.size": FONT_SIZES.get("small"),
    "axes.titlesize": FONT_SIZES.get("small"),
    "axes.labelsize": FONT_SIZES.get("medium"),
    "xtick.labelsize": FONT_SIZES.get("small"),
    "ytick.labelsize": FONT_SIZES.get("small"),
    "legend.fontsize": FONT_SIZES.get("small"),
    "figure.titlesize": FONT_SIZES.get("medium"),
}
if _get_font():
    PLOT_PARAMS["font.family"] = _get_font()
plt.rcParams.update(PLOT_PARAMS)


def draw_per_token_reward(
    tokens: List[str],
    rewards: List[List[float]],
    model_names: List[str],
    font_size: int = 12,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 12),
    line_chart: bool = False,
) -> "matplotlib.axes.Axes":
    """Draw a heatmap that combines the rewards

    tokens (List[str]): the canonical tokens that was used as reference during alignment.
    rewards (List[List[float]]): list of rewards-per-token for each model.
    model_names (List[str]): list of models.
    font_size (int): set the font size.
    output_path (Optional[Path]): if set, then save the figure in the specified path.
    figsize (Tuple[int, int]): control the figure size when plotting.
    line_chart (bool): if set, will draw a line chart instead of a figure.
    RETURNS (matplotlib.axes.Axes): an Axes class containing the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rewards = np.array(rewards)
    if not line_chart:
        im = ax.imshow(
            rewards,
            cmap="viridis",
            vmax=np.max(rewards),
            vmin=np.min(rewards),
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", aspect=20, location="bottom")
        ax.set_xticks(np.arange(len(tokens)), [f'"{token}"' for token in tokens])
        ax.set_yticks(np.arange(len(model_names)), model_names)

        # Add text
        avg = np.mean(rewards)
        for i in range(len(model_names)):
            for j in range(len(tokens)):
                color = "k" if rewards[i, j] >= avg else "w"
                ax.text(j, i, round(rewards[i, j], 4), ha="center", va="center", color=color)

        # Make it look better
        ax.xaxis.tick_top()
        ax.tick_params(left=False, top=False)
        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    else:
        print("Drawing line chart")
        idxs = np.arange(0, len(tokens))
        for model_name, per_token_rewards in zip(model_names, rewards):
            ax.plot(idxs, per_token_rewards, label=model_name, marker="x")

        ax.legend(loc="upper left")
        ax.set_xticks(np.arange(len(tokens)), [f'"{token}"' for token in tokens])
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Reward")
        ax.spines[["right", "top"]].set_visible(False)

    # Added information
    title = "Cumulative substring rewards"
    ax.set_title(title, pad=20)

    # fig.tight_layout()
    if not line_chart:
        fig.subplots_adjust(left=0.5)
    if output_path:
        print(f"Saving per-token-reward plot to {output_path}")
        plt.savefig(output_path, transparent=True, dpi=120)

    plt.show()


def print_model_statistics(
    dataset_name: str = "allenai/reward-bench",
    keys: List[str] = ["chosen_model", "rejected_model"],
    render_latex: bool = False,
):
    """Print model counts and statistics into a Markdown/LaTeX table

    dataset_name (str): the HuggingFace dataset name to source the eval dataset.
    keys (List[str]): the dataset columns to include in the histogram.
    render_latex (bool): if True, render a LaTeX string.
    RETURNS (str): a Markdown/LaTeX rendering of a table.
    """
    dataset = datasets.load_dataset(dataset_name, split="filtered")

    models = {key: [] for key in keys}
    for example in dataset:
        for key in keys:
            model = example[key]
            if model == "unkown":
                # Fix: https://huggingface.co/datasets/allenai/reward-bench/discussions/1
                model = "unknown"
            models[key].append(model)
    counters = [Counter(models) for key, models in models.items()]

    # create another counter which is the sum of all in counters
    total_ctr = sum(counters, Counter())
    # create a table with model, total counter,
    # and the other counters by keys (0 if not in the sub counter)
    total_df = pd.DataFrame(total_ctr.most_common(), columns=["Model", "Total"])
    chosen_ctr, rejected_ctr = counters
    chosen_df = pd.DataFrame(chosen_ctr.most_common(), columns=["Model", "chosen_model"])
    rejected_df = pd.DataFrame(rejected_ctr.most_common(), columns=["Model", "rejected_model"])
    # merge these DataFrames into a single value
    model_statistics_df = (
        total_df.merge(chosen_df, how="left")
        .merge(rejected_df, how="left")
        .fillna(0)
        .astype({key: int for key in keys})
    )

    render_string = (
        model_statistics_df.to_latex(index=False)
        if render_latex
        else model_statistics_df.to_markdown(index=False, tablefmt="github")
    )
    print(render_string)
    print(f"\nTotal number of models involved: {len(total_ctr) - 2}")
    return render_string


def draw_model_source_histogram(
    dataset_name: str = "allenai/reward-bench",
    output_path: Optional[str] = None,
    keys: List[str] = ["chosen_model", "rejected_model"],
    figsize: Tuple[int, int] = (8, 4),
    font_size: int = 15,
    normalize: bool = False,
    log_scale: bool = False,
    include_title: bool = False,
    top_n: Optional[int] = None,
) -> "matplotlib.axes.Axes":
    """Draw a histogram of the evaluation dataset that shows completion counts between models and humans.

    dataset_name (str): the HuggingFace dataset name to source the eval dataset.
    output_path (Optional[Path]): if set, then save the figure in the specified path.
    keys (List[str]): the dataset columns to include in the histogram.
    figsize (Tuple[int, int]): control the figure size when plotting.
    font_size (int): set the font size.
    normalize (bool): set to True to normalize the values based on total number completions.
    log_scale (bool): set the y-axis to logarithmic scale.
    top_n (Optional[int]): if set, then only plot the top-n models in the histogram.
    include_title (bool): if set, then will include the title in the chart.
    RETURNS (matplotlib.axes.Axes): an Axes class containing the histogram.
    """
    dataset = datasets.load_dataset(dataset_name, split="filtered")

    if not all(key in dataset.features for key in keys):
        raise ValueError(f"Your dataset has missing keys. Please ensure that {keys} is/are available.")

    models = []
    for example in dataset:
        for key in keys:
            model = example[key]
            if model == "unkown":
                # Fix: https://huggingface.co/datasets/allenai/reward-bench/discussions/1
                model = "unknown"
            models.append(model)
    counter = Counter(models)

    if normalize:
        total = sum(counter.values(), 0.0)
        for key in counter:
            counter[key] /= total

    # Draw the histogram
    fig, ax = plt.subplots(figsize=figsize)
    labels, values = zip(*counter.most_common())

    if top_n:
        labels = labels[:top_n]
        values = values[:top_n]

    indices = list(reversed(np.arange(len(labels))))
    width = 1

    colors = [AI2_COLORS.get("light_blue"), AI2_COLORS.get("light_aqua")]
    ax.barh(indices, values, width, color=colors * (len(indices) // 2 + 1))
    # ax.set_xticks(indices, labels, rotation=90)
    ax.set_yticks(indices, labels)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Source of completion")
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    # plt.margins(0, 0.05)
    plt.margins(0.05, 0)

    title = f"Source of completions ({', '.join([k.replace('_', ' ') for k in keys])})"

    if normalize:
        ax.set_ylim(top=1.00)
        title += " , normalized"

    if log_scale:
        ax.set_xscale("log")
        title += ", log-scale"

    if top_n:
        title += f", showing top-{top_n}"

    if include_title:
        ax.set_title(title)
    fig.tight_layout()

    if output_path:
        print(f"Saving histogram to {output_path}")
        plt.savefig(output_path, transparent=True, dpi=120)

    return ax


def draw_subtoken_statistics(
    category_subsets: Dict[str, List[str]],
    output_path: Optional[Path] = None,
    dataset_name: str = "allenai/reward-bench",
    tokenizer_name: str = "oobabooga/llama-tokenizer",
    figsize: Tuple[int, int] = (8, 4),
    render_latex: bool = False,
) -> Tuple["matplotlib.axes.Axes", "pd.DataFrame"]:
    subsets = get_dataset_tokens_per_subset(
        tokenizer_name=tokenizer_name,
        dataset_name=dataset_name,
        split="filtered",
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

    def _get_category(name: str):
        for category, subsets in category_subsets.items():
            if name in subsets:
                return category

    # Create report table
    df = pd.DataFrame(
        [
            {
                "category": _get_category(name),
                "subset": SUBSET_NAME_TO_PAPER_READY[name],
                "chosen_avg": np.mean(stats["chosen_lens"]),
                "chosen_max": np.max(stats["chosen_lens"]),
                "chosen_min": np.min(stats["chosen_lens"]),
                "chosen_std": np.std(stats["chosen_lens"]),
                "chosen_unique_avg": np.mean(stats["chosen_unique_lens"]),
                "chosen_unique_max": np.max(stats["chosen_unique_lens"]),
                "chosen_unique_min": np.min(stats["chosen_unique_lens"]),
                "rejected_avg": np.mean(stats["rejected_lens"]),
                "rejected_max": np.max(stats["rejected_lens"]),
                "rejected_min": np.min(stats["rejected_lens"]),
                "rejected_std": np.std(stats["rejected_lens"]),
                "rejected_unique_avg": np.mean(stats["rejected_unique_lens"]),
                "rejected_unique_max": np.max(stats["rejected_unique_lens"]),
                "rejected_unique_min": np.min(stats["rejected_unique_lens"]),
            }
            for name, stats in subtoken_statistics.items()
        ]
    )

    df = df.sort_values(by=["category", "subset"]).reset_index(drop=True)
    render_string = (
        df.round(4).astype(str).to_latex(index=False)
        if render_latex
        else df.to_markdown(index=False, tablefmt="github")
    )
    render_string = render_string.replace("NaN", "")
    render_string = render_string.replace("nan", "")
    print(render_string)

    # Plotting
    # n_categories = df["category"].nunique()
    # fig, ax = plt.subplots(figsize=figsize)
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axs = np.ravel(axs)
    for ax, (category, df) in zip(axs, df.groupby("category")):
        labels = df["subset"].to_list()
        chosen_avgs = df["chosen_avg"].to_list()
        chosen_stds = df["chosen_std"].to_list()
        rejected_avgs = df["rejected_avg"].to_list()
        rejected_stds = df["rejected_std"].to_list()
        indices = list(reversed(np.arange(0, len(labels))))
        # Chosen stats
        ax.errorbar(
            chosen_avgs,
            indices,
            xerr=chosen_stds,
            color=AI2_COLORS.get("light_blue"),
            fmt="o",
            elinewidth=2,
            capsize=2,
            markersize=10,
            label="Chosen",
        )
        # Rejected stats
        ax.errorbar(
            rejected_avgs,
            indices,
            xerr=rejected_stds,
            color=AI2_COLORS.get("light_red"),
            fmt="o",
            markersize=10,
            elinewidth=2,
            capsize=2,
            label="Rejected",
        )

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_yticks(indices, labels)
        ax.set_title(category)
        ax.set_xlim([0, 1000])
        ax.set_xlabel("Prompt length")

    # Assign everything to last
    # axs[2].legend(loc=(1.0, -0.55), frameon=False, ncol=2)
    # ax.set_xlabel("Prompt length")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    if output_path:
        print(f"Saving to {output_path}")
        plt.savefig(output_path, transparent=True, dpi=120, bbox_inches="tight")

    return ax, df


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
