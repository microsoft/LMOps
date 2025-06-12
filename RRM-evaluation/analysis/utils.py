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

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_scores(
    repo_dir_path: Union[str, Path],
    subdir: str,
    # ignore_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load results into a pandas DataFrame"""
    base_dir = Path(repo_dir_path)
    data_dir = base_dir / subdir
    orgs_dir = {d.name: d for d in data_dir.iterdir() if d.is_dir()}
    # Get all files within the subfolder orgs
    model_result_files = {d: list(path.glob("*.json")) for d, path in orgs_dir.items()}

    _results: List[pd.DataFrame] = []  # will merge later
    for org, filepaths in model_result_files.items():
        for filepath in filepaths:
            if "nfs.cirrascale" not in str(filepath).split("scores/")[-1]:  # ignore internal ai2 data
                _results.append(pd.read_json(filepath, orient="records"))
    results_df = pd.concat(_results)
    return results_df


def load_results(
    repo_dir_path: Union[str, Path],
    subdir: str,
    ignore_columns: Optional[List[str]] = None,
    filepath_filter: Optional[str] = None,
    remove_ref_free: bool = True,
) -> pd.DataFrame:
    """Load results into a pandas DataFrame"""
    base_dir = Path(repo_dir_path)
    data_dir = base_dir / subdir
    orgs_dir = {d.name: d for d in data_dir.iterdir() if d.is_dir()}
    # Get all files within the subfolder orgs
    model_result_files = {d: list(path.glob("*.json")) for d, path in orgs_dir.items()}

    _results: List[pd.DataFrame] = []  # will merge later
    for org, filepaths in model_result_files.items():
        for filepath in filepaths:
            # optionally filter to only files including a specific string
            if filepath_filter is not None:
                if filepath_filter not in str(filepath):
                    continue
            _results.append(pd.DataFrame(load_dataset("json", data_files=str(filepath), split="train")))
    results_df = pd.concat(_results)

    # remove internal experiments under org ai2
    results_df = results_df[~results_df["model"].str.contains("ai2")]

    # Cleanup the dataframe for presentation
    def _cleanup(df: pd.DataFrame) -> pd.DataFrame:
        # remove chat_template comlumn
        df = df.drop(columns=["chat_template"])

        # sort columns alphabetically
        df = df.reindex(sorted(df.columns), axis=1)

        # move column "model" to the front
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index("model")))
        df = df.loc[:, cols]

        # select all columns except "model"
        cols = df.columns.tolist()
        cols.remove("model")
        # if model_type is a column (pref tests may not have it)
        if "model_type" in cols:
            cols.remove("model_type")
        # remove model_beaker from dataframe
        if "model_beaker" in cols:
            cols.remove("model_beaker")
            df = df.drop(columns=["model_beaker"])

        # remove ref_model
        if "ref_model" in cols:
            cols.remove("ref_model")
            df = df.drop(columns=["ref_model"])

        if "xstest" in cols:
            cols.remove("xstest")
            df = df.drop(columns=["xstest"])

        # remove column anthropic and summarize_prompted (outdated data)
        if "anthropic" in cols:
            df = df.drop(columns=["anthropic"])
            cols.remove("anthropic")
        if "summarize_prompted" in cols:
            df = df.drop(columns=["summarize_prompted"])
            cols.remove("summarize_prompted")
        # remove pku_better and pku_safer (removed from the leaderboard)
        if "pku_better" in cols:
            df = df.drop(columns=["pku_better"])
            cols.remove("pku_better")
        if "pku_safer" in cols:
            df = df.drop(columns=["pku_safer"])
            cols.remove("pku_safer")

        # round
        df[cols] = df[cols]
        avg = np.nanmean(df[cols].values, axis=1)
        # add average column
        df["average"] = avg

        # move average column to the second
        cols = list(df.columns)
        cols.insert(1, cols.pop(cols.index("average")))
        df = df.loc[:, cols]

        if "model_type" in cols:
            # get model_types that have generative in them
            mask = df["model_type"].str.contains("generative", case=False, na=False)

            # set these values to "Generative"
            df.loc[mask, "model_type"] = "Generative"

            cols = list(df.columns)
            cols.insert(1, cols.pop(cols.index("model_type")))
            df = df.loc[:, cols]

        # remove models with DPO Ref. Free as type (future work)
        if remove_ref_free:
            df = df[~df["model_type"].str.contains("DPO Ref. Free", na=False)]

        # remove columns
        if ignore_columns:
            # Get columns from df that exist in ignore_columns
            _ignore_columns = [col for col in ignore_columns if col in df.columns]
            if len(_ignore_columns) > 0:
                print(f"Dropping columns: {', '.join(_ignore_columns)}")
                df = df.drop(_ignore_columns, axis=1)

        return df

    results_df = _cleanup(results_df)
    return results_df
