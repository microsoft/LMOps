# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import csv
import dataclasses
import os
import pathlib
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import FullDatum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.eval import Metric
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.search import PartialParse


@dataclass(frozen=True)
class BreakDatum(FullDatum):
    # Corresponds to the "allowed words" lexicon for a given datum in the Break dataset
    allowed_tokens: Set[int]

    # Corresponds to the original Break decomposition, different from our internal decomposition format.
    decomposition: str

    # If the datum is in train/test/validation.
    split: str


class BreakDataType(str, Enum):
    qdmr = "QDMR"
    nested = "nested"


def parse_allowed_tokens(allowed_tokens_str: str) -> Set[str]:
    return {
        str(s).strip()
        for s in ast.literal_eval(allowed_tokens_str)
        if not str(s).startswith("@@")
    }


class BreakSamplingType(Enum):
    equal = "equal"
    proportional = "proportional"
    random = "random"


def can_close_paren(s: str) -> bool:
    if len(s) == 0:
        return False

    return s.count("(") > s.count(")")


@dataclass
class BreakCommonTokens:
    tokenizer: GPT2Tokenizer
    open_paren: int
    close_paren: int
    return_: int
    return_without_space: int
    semicolon: int
    pound_sign: int
    numbers: List[List[int]]

    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer):
        [open_paren] = tokenizer.encode(" (", add_special_tokens=False)
        [close_paren] = tokenizer.encode(" )", add_special_tokens=False)
        [return_with_space] = tokenizer.encode(" return", add_special_tokens=False)
        [return_without_space] = tokenizer.encode("return", add_special_tokens=False)
        [semicolon] = tokenizer.encode(" ;", add_special_tokens=False)
        [pound_sign] = tokenizer.encode(" #", add_special_tokens=False)

        numbers = [
            tokenizer.encode(str(i), add_special_tokens=False) for i in range(21)
        ]
        return BreakCommonTokens(
            tokenizer,
            open_paren,
            close_paren,
            return_with_space,
            return_without_space,
            semicolon,
            pound_sign,
            numbers,
        )


@dataclass
class BreakPartialParse(PartialParse):
    common_tokens: BreakCommonTokens
    allowed_tokens_with_space: Set[int]
    allowed_tokens: Set[int]
    data_type: BreakDataType
    prefix: str = ""

    # datum: BreakDatum

    @classmethod
    def initial(
        cls,
        common_tokens: BreakCommonTokens,
        data_type: BreakDataType,
        datum: BreakDatum,
    ) -> "BreakPartialParse":
        allowed_tokens_with_space = set()
        allowed_tokens = set()

        space_char = common_tokens.tokenizer.byte_encoder[ord(" ")]
        for token_id in datum.allowed_tokens:
            if common_tokens.tokenizer.decoder[token_id][0] == space_char:
                allowed_tokens_with_space.add(token_id)
            allowed_tokens.add(token_id)

        return cls(common_tokens, allowed_tokens_with_space, allowed_tokens, data_type)

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, bool]:
        if self.data_type == BreakDataType.nested:
            return self._allowed_next_nested()
        else:
            return self._allowed_next_qdmr()

    def _allowed_next_nested(self) -> Tuple[torch.Tensor, bool]:
        is_balanced, max_depth = BreakPieces.is_balanced(self.prefix)

        is_eos = len(self.prefix) > 0 and is_balanced

        open_parens = (
            {self.common_tokens.open_paren} if not self.prefix.endswith(")") else set()
        )
        close_parens = (
            {self.common_tokens.close_paren} if can_close_paren(self.prefix) else set()
        )

        if not is_eos and max_depth >= 10:
            next_tokens = self._currently_allowed_tokens | close_parens
        else:
            next_tokens = self._currently_allowed_tokens | open_parens | close_parens

        return torch.tensor(list(next_tokens), dtype=torch.long), is_eos

    def _allowed_next_qdmr(self) -> Tuple[torch.Tensor, bool]:
        num_steps = self.prefix.count(";")

        references = {
            t for i in range(1, num_steps + 1) for t in self.common_tokens.numbers[i]
        }
        ends_with_semicolon = self.prefix.endswith(";")
        ends_with_reference = self.prefix.endswith("#")

        next_tokens: Set[int]

        if len(self.prefix) == 0:
            next_tokens = {self.common_tokens.return_}
        elif ends_with_semicolon:
            next_tokens = {self.common_tokens.return_without_space}
        elif ends_with_reference:
            next_tokens = references
        elif num_steps > 20:
            next_tokens = {self.common_tokens.semicolon}
        else:
            next_tokens = (
                self._currently_allowed_tokens
                | references
                | {self.common_tokens.semicolon}
                | {self.common_tokens.pound_sign}
            )

        is_eos = len(self.prefix) > 0
        return torch.tensor(list(next_tokens), dtype=torch.long), is_eos

    def append(self, token: int) -> "PartialParse":
        """Return a new PartialParse creatoted by appending this token."""
        return dataclasses.replace(
            self,
            prefix=self.prefix
            + self.common_tokens.tokenizer.convert_tokens_to_string(
                self.common_tokens.tokenizer.decoder[token]
            ),
        )

    @property
    def _currently_allowed_tokens(self):
        if self.prefix == "":
            return self.allowed_tokens_with_space
        else:
            return self.allowed_tokens


@dataclass
class BreakPieces:
    train_data: List[BreakDatum]
    test_data: List[BreakDatum]
    partial_parse_builder: Callable[[BreakDatum], BreakPartialParse]
    data_type: BreakDataType

    @staticmethod
    def simplify(decomposition_str: str) -> str:
        return re.sub(" +", " ", decomposition_str)

    @staticmethod
    def nest(decomposition_str: str) -> str:
        steps = decomposition_str.split(";")

        for i, step in enumerate(steps):
            steps[i] = re.sub(" +", " ", step[len("return ") :].strip())

            for j in range(i, len(steps)):
                steps[j] = steps[j].replace(f"#{i+1} ", f"( {steps[i]} ) ")
                steps[j] = re.sub(f"#{i+1}$", f"( {steps[i]} )", steps[j])

        return steps[-1]

    @staticmethod
    def unnest(nested_str: str) -> str:
        stack = []

        levels: Dict[int, Dict[str, Optional[str]]] = defaultdict(dict)

        for i, c in enumerate(nested_str):
            if c == "(":
                stack.append(i)
            elif c == ")" and len(stack) > 0:
                start = stack.pop()
                levels[len(stack) + 1][nested_str[start + 1 : i]] = None

        steps = [
            step
            for level, level_dict in sorted(levels.items(), reverse=True)
            for step in level_dict.keys()
        ] + [nested_str]

        for i, step in enumerate(steps):
            steps[i] = step.strip()
            for j in range(i, len(steps)):
                steps[j] = steps[j].replace(f"( {steps[i]} )", f"#{i+1}")

            steps[i] = "return " + steps[i]

        return " ;".join(steps)

    @staticmethod
    def simplify_nesting(s: str) -> str:
        prev_s = s

        s = re.sub(r"\( \(([^\(\)]+)\) \)", r"(\1)", s)

        while s != prev_s:
            prev_s = s
            s = re.sub(r"\( \(([^\(\)]+)\) \)", r"(\1)", s)

        possible_s = re.sub(r"^\( (.+) \)$", r"\1", s)

        return possible_s if BreakPieces.is_balanced(possible_s)[0] else s

    @staticmethod
    def is_balanced(s: str) -> Tuple[bool, int]:
        count = 0
        max_depth = 0
        for c in s:
            if c == "(":
                count += 1
                max_depth = max(max_depth, count)
            if c == ")":
                count -= 1

            if count < 0:
                return False, max_depth

        return count == 0, max_depth

    @staticmethod
    def build(
        tokenizer: GPT2Tokenizer,
        data_type: BreakDataType,
        train_sampling_type: BreakSamplingType,
        test_sampling_type: BreakSamplingType,
        train_total: int,
        test_total: int,
        seed: int,
        skip_if_needed: bool = True,
    ) -> "BreakPieces":
        dataset = load_dataset("break_data", "QDMR")
        train_data = dataset["train"]
        dev_data = dataset["validation"]

        lexicon = load_dataset("break_data", "QDMR-lexicon")
        train_allowed_tokens = {
            d["source"]: d["allowed_tokens"] for d in lexicon["train"]
        }
        dev_allowed_tokens = {
            d["source"]: d["allowed_tokens"] for d in lexicon["validation"]
        }

        common_tokens = BreakCommonTokens.from_tokenizer(tokenizer)
        partial_parse_builder = lambda datum: BreakPartialParse.initial(
            common_tokens, data_type, datum
        )

        train_break_data = BreakPieces.generate_break_data(
            data_type,
            tokenizer,
            train_allowed_tokens,
            train_data,
            train_sampling_type,
            train_total,
            skip_unparsable=skip_if_needed,
        )
        if skip_if_needed:
            test_break_data = BreakPieces.generate_break_data(
                data_type,
                tokenizer,
                dev_allowed_tokens,
                dev_data,
                test_sampling_type,
                2 * test_total,
                skip_unparsable=skip_if_needed,
            )[
                test_total:
            ]  # so sad, this way we avoid the first batch where we peeked at the dev data
        else:
            test_break_data = BreakPieces.generate_break_data(
                data_type,
                tokenizer,
                dev_allowed_tokens,
                dev_data,
                test_sampling_type,
                test_total,
                skip_unparsable=skip_if_needed,
            )

        random.Random(seed).shuffle(train_break_data)
        random.Random(seed).shuffle(test_break_data)

        return BreakPieces(
            train_break_data, test_break_data, partial_parse_builder, data_type
        )

    @staticmethod
    def generate_break_data(
        data_type: BreakDataType,
        tokenizer: GPT2Tokenizer,
        allowed_tokens_dict,
        all_data,
        sampling_type,
        total,
        skip_unparsable: bool,
    ):
        all_count = {}
        count: Dict[str, int] = defaultdict(int)
        unparsable_count: Dict[str, int] = defaultdict(int)
        break_data = []

        grouped_data = {
            dataset_type: list(group_iter)
            for dataset_type, group_iter in groupby(
                sorted(all_data, key=lambda d: d["question_id"]),
                lambda d: "all"
                if sampling_type == BreakSamplingType.random
                else d["question_id"].split("_")[0],
            )
        }

        for group, data in grouped_data.items():
            random.Random(0).shuffle(data)
            all_count[group] = len(data)

            for d in data:
                if sampling_type == BreakSamplingType.equal and count[
                    group
                ] >= total / len(grouped_data):
                    break

                if sampling_type == BreakSamplingType.proportional and count[
                    group
                ] >= total * len(data) / len(all_data):
                    break

                if sampling_type == BreakSamplingType.random and count[group] >= total:
                    break

                parsed = (
                    BreakPieces.nest(d["decomposition"])
                    if data_type == BreakDataType.nested
                    else BreakPieces.simplify(d["decomposition"])
                )

                allowed_tokens = allowed_tokens_dict.get(d["question_text"])

                if skip_unparsable and (allowed_tokens is None or parsed is None):
                    unparsable_count[group] += 1
                    continue

                valid_tokens = parse_allowed_tokens(allowed_tokens)
                copy_tokens = set()

                for s in valid_tokens:
                    copy_tokens.update(tokenizer.encode(s, add_special_tokens=False))
                    copy_tokens.update(
                        tokenizer.encode(" " + s, add_special_tokens=False)
                    )

                count[group] += 1
                break_data.append(
                    BreakDatum(
                        dialogue_id=d["question_id"],
                        natural=d["question_text"],
                        decomposition=d["decomposition"],
                        canonical=parsed,
                        allowed_tokens=copy_tokens,
                        turn_part_index=None,
                        agent_context=None,
                        split=d["split"],
                    )
                )

        print("Sampling type: ", sampling_type)
        print("All count: ", dict(all_count))
        print("Total all count: ", sum(all_count.values()))

        print("Unparsable count: ", dict(unparsable_count))
        print("Total unparsable count: ", sum(unparsable_count.values()))

        print("Kept count: ", dict(count))
        print("Total kept count: ", sum(count.values()))
        print("\n")

        return break_data


@dataclass
class BreakResponse:
    question_id: str
    question_text: str
    decomposition: str
    split: str
    pred: str
    k: int

    def response_id(self) -> str:
        return f"{self.question_id}@{self.k}"


@dataclass
class BreakMetrics(Metric[Sequence[str], BreakDatum]):
    log_dir: pathlib.Path
    data_type: BreakDataType
    # The beam size, or the number of predictions we should expect for each datum
    num_results: int

    rows: List[BreakResponse] = dataclasses.field(default_factory=list)

    def parsed_to_decomposition(self, s: str) -> str:
        if self.data_type == BreakDataType.nested:
            return BreakPieces.unnest(BreakPieces.simplify_nesting(s))
        else:
            return s

    def update(
        self, preds: Sequence[str], target: BreakDatum
    ) -> Dict[str, Optional[str]]:
        if len(preds) == 0:
            preds = ["return "]
        for i, pred in enumerate(preds):
            self.rows.append(
                BreakResponse(
                    target.dialogue_id,
                    target.natural,
                    target.decomposition,
                    target.split,
                    self.parsed_to_decomposition(pred),
                    i,
                )
            )

        return {}

    def compute(self) -> Dict[str, float]:
        os.makedirs(self.log_dir, exist_ok=True)
        decomp_csv_path = self.log_dir / "decomp.csv"
        label_csv_path = self.log_dir / "label.csv"
        decomp_with_id_csv_path = self.log_dir / "decomp_with_id.csv"

        with open(decomp_csv_path, "w", newline="") as decomp_file, open(
            label_csv_path, "w", newline=""
        ) as labels, open(decomp_with_id_csv_path, "w", newline="") as decomp_with_id:
            writer = csv.DictWriter(
                labels,
                fieldnames=["question_id", "question_text", "decomposition", "split"],
            )
            writer.writeheader()

            for t in self.rows:
                writer.writerow(
                    {
                        "question_id": t.response_id(),
                        "question_text": t.question_text,
                        "decomposition": t.decomposition,
                        "split": t.split,
                    }
                )

            labels.flush()

            out = csv.writer(decomp_file)
            out.writerow(["decomposition"])
            for r in self.rows:
                out.writerow([r.pred])
            decomp_file.flush()

            out_with_id = csv.writer(decomp_with_id)
            out_with_id.writerow(["question_id", "decomposition"])
            for r in self.rows:
                out_with_id.writerow([r.response_id(), r.pred])
            decomp_with_id.flush()

            # To install:
            # - Add your SSH key to https://dev.azure.com/semanticmachines/_usersSettings/keys
            # - Run: pipx install --python python3.7 git+ssh://git@ssh.dev.azure.com/v3/semanticmachines/SemanticMachines/break-evaluator
            subprocess.run(
                [
                    "break_evaluate_predictions",
                    "--dataset_file",
                    label_csv_path,
                    "--preds_file",
                    decomp_file.name,
                    "--no_cache",
                    "--output_file_base",
                    str(self.log_dir.absolute()) + "/",
                    "--metrics",
                    "exact_match",
                    "normalized_exact_match",
                ],
                check=True,
            )

            summary = csv.DictReader(
                open(f"{self.log_dir}/_summary.tsv"), delimiter="\t"
            )

            # example id => List[em, nem] for each prediction
            results: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)

            for row in summary:
                [example_id, _] = row["id"].split("@")
                exact_match = row["exact_match"] == "True"
                normalized_exact_match = row["normalized_exact_match"] == "True"
                results[example_id].append((exact_match, normalized_exact_match))

            correct_em_counts: Dict[int, int] = defaultdict(int)
            correct_nem_counts: Dict[int, int] = defaultdict(int)
            total = 0

            for example_id, corrects in results.items():
                total += 1
                found_em_correct = False
                found_nem_correct = False

                for i, (em_correct, nem_correct) in enumerate(corrects):
                    found_em_correct |= em_correct
                    correct_em_counts[i] += found_em_correct

                    found_nem_correct |= nem_correct
                    correct_nem_counts[i] += found_nem_correct

                # Handle when we have fewer predictions than self.num_results
                for i in range(len(corrects), self.num_results):
                    correct_em_counts[i] += found_em_correct
                    correct_nem_counts[i] += found_nem_correct

            metrics = {}
            for i in range(self.num_results):
                metrics[f"nem @ {i + 1}"] = correct_nem_counts[i] / total
                metrics[f"nem count @ {i + 1}"] = correct_nem_counts[i]

            for i in range(self.num_results):
                metrics[f"em @ {i + 1}"] = correct_em_counts[i] / total
                metrics[f"em count @ {i + 1}"] = correct_em_counts[i]
            metrics["total"] = total

            return metrics

    def reset(self) -> None:
        self.rows = []
