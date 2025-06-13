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

# Script for aggregating previous scores via ensemble to explore RM ensemble performance


import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section


def get_args():
    """
    Argparser. Gets the models you wish to analyze primarily.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="allenai/reward-bench-results",
        help="HuggingFace repository containing the evaluation results.",
    )
    parser.add_argument("--models", type=str, nargs="+", help="Models to analyze.")
    parser.add_argument("--do_not_normalize", action="store_true", default=False, help="Do not normalize the values.")
    # mode is ether Mean, Worst, or Uncertainty
    parser.add_argument("--mode", type=str, default="Mean", help="Mode of aggregation.")
    parser.add_argument("--pref_sets", action="store_true", help="Use preference sets.")
    parser.add_argument("--sweep", action="store_true", default=False, help="Sweep over all model options from >3.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    all_models = args.models

    #########################
    # Setup and Load
    #########################
    assert isinstance(all_models, list), "Models must be a list."
    assert len(all_models) > 1, "Models must not alone."

    # Assert that modes are valid
    assert args.mode in ["Mean", "Worst", "Uncertainty"], "Invalid mode."

    # Load the results for the models
    subdir = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"
    baseline_values = {}
    data = {}

    def flatten(data):
        # if all rewards is list of list, unnest
        if isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]
        return data

    for m in all_models:
        hub_file = subdir + f"{m}.json"
        f = hf_hub_download(args.hf_evals_repo, hub_file, repo_type="dataset")
        eval_data = pd.read_json(f, orient="records")

        # add baseline values for each model
        all_rewards = np.concatenate((eval_data["scores_rejected"].values, eval_data["scores_chosen"]))
        all_rewards = flatten(all_rewards)
        mean_reward = np.mean(all_rewards)
        std_dev_reward = np.std(all_rewards)
        baseline_values[m] = {"mean": mean_reward, "std_dev": std_dev_reward}

        data[m] = eval_data

    #########################
    # Normalize
    #########################
    if not args.do_not_normalize:
        for m in all_models:
            data[m]["scores_rejected"] = (
                flatten(data[m]["scores_rejected"]) - baseline_values[m]["mean"]
            ) / baseline_values[m]["std_dev"]
            data[m]["scores_chosen"] = (
                flatten(data[m]["scores_chosen"]) - baseline_values[m]["mean"]
            ) / baseline_values[m]["std_dev"]

    print(f"All models: {all_models}")
    all_results = []

    # check if sweep
    if args.sweep:
        modes = ["Mean", "Worst", "Uncertainty"]
        model_index = 2
    else:
        modes = [args.mode]
        model_index = len(all_models)

    # iterate over all subsets from length 3 to 6 models
    from itertools import combinations

    for mode in modes:
        args.mode = mode
        for i in range(model_index, len(all_models) + 1):
            for models in combinations(all_models, i):
                models = list(models)

                print(f"Analyzing models: {models}")

                #########################
                # Calculate ensembles
                #########################
                def compute_reward(scores, mode):
                    if mode == "Mean":
                        return np.mean(scores)
                    elif mode == "Worst":
                        return np.min(scores)
                    elif mode == "Uncertainty":
                        return np.mean(scores) - np.std(scores)

                # iterate over ids in the dataframe
                ids = data[models[0]]["id"].unique()
                out_dataset = {"subsets": [], "results": []}
                for id in ids:
                    scores_chosen = []
                    scores_rejected = []
                    for m in models:
                        scores_chosen.append(data[m].loc[data[m]["id"] == id]["scores_chosen"].values[0])
                        scores_rejected.append(data[m].loc[data[m]["id"] == id]["scores_rejected"].values[0])

                    ensemble_score_chosen = compute_reward(np.array(scores_chosen), args.mode)
                    ensemble_score_rejected = compute_reward(np.array(scores_rejected), args.mode)
                    subset = data[models[0]].loc[data[models[0]]["id"] == id]["subset"].values[0]
                    out_dataset["subsets"].append(subset)
                    value = 1 if ensemble_score_chosen > ensemble_score_rejected else 0
                    out_dataset["results"].append(value)

                out_dataset = Dataset.from_dict(out_dataset).to_pandas()  # I know this is meh

                #########################
                # Save / Share
                #########################

                results_grouped = {}
                present_subsets = np.unique(out_dataset["subsets"])
                for subset in present_subsets:
                    # subset_dataset = out_dataset.filter(lambda example: example["subsets"] == subset)
                    subset_dataset = out_dataset[out_dataset["subsets"] == subset]
                    num_correct = sum(subset_dataset["results"])
                    num_total = len(subset_dataset["results"])
                    # print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
                    results_grouped[subset] = num_correct / num_total

                if not args.pref_sets:
                    results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
                    print(results_leaderboard)
                    results_leaderboard["models"] = "|".join(models)
                    results_leaderboard["mode"] = args.mode
                    all_results.append(results_leaderboard)

    all_results = Dataset.from_list(all_results)
    all_results.to_csv("ensemble_results.csv")
