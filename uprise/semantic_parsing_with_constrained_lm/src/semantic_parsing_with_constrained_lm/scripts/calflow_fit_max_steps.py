# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Computes an linear fit for the number of tokens in the input and output.

This can be used to set a good parameters for the `max_steps` parameter in beam search.

The script does 10-fold cross-validation to find a slope and intercept which:
- minimizes the mean number of excess steps (i.e. predicted length - gold length)
- only very rarely predicts a length which is smaller than the gold length

Example invocation:
python semantic_parsing_with_constrained_lm/scripts/calflow_fit_max_steps.py \
    --data-path semantic_parsing_with_constrained_lm/domains/calflow/data/train_300_stratified.jsonl \
    --tokenizer facebook/bart-large \
    --output-type canonicalUtterance  # or "lispress"
"""
import pathlib
from typing import List, Tuple

import typer
from transformers import AutoTokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow import (
    CalflowOutputLanguage,
    read_calflow_jsonl,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.fit_max_steps import compute_and_print_fit


def main(
    data_path: pathlib.Path = typer.Option(...),
    tokenizer: str = typer.Option(...),
    output_type: CalflowOutputLanguage = typer.Option(...),
    max_unreachable: int = typer.Option(1),
):
    t = AutoTokenizer.from_pretrained(tokenizer)

    pairs: List[Tuple[int, int]] = []
    for datum in read_calflow_jsonl(data_path, output_type):
        num_input_tokens = len(t.tokenize(datum.natural))
        if not datum.canonical:
            continue
        num_output_tokens = len(t.tokenize(datum.canonical)) + 1

        pairs.append((num_input_tokens, num_output_tokens))

    compute_and_print_fit(pairs, 10, max_unreachable)


if __name__ == "__main__":
    typer.run(main)
