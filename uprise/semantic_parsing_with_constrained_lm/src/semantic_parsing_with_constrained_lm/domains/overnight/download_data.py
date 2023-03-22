# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Downloads data for Overnight from the public CodaLab instance, and prepares it into json/jsonl formats.

There are eight domains:
- housing
- calendar
- socialnetwork
- basketball
- blocks
- publications
- recipes
- restaurants

For each domain, we output the following files:
- [domain name].train_with_dev.jsonl
- [domain name].train_without_dev.jsonl
- [domain name].dev.jsonl
- [domain name].test.jsonl
Each line of these JSONL files contains the following keys:
- canonical: a string containing the synthetic utterance
- natural: a string containing the paraphrased utterance
- formula: a string containing an S-expression

We also output [domain name].canonical.json, which contains a single object (a map)
with canonical utterances as keys, and objects with the following keys as values:
- paraphrases: an array of strings, containing all paraphrases collected from crowd workers
- formula: a string containing an S-expression
- denotation: a string containing the execution results of the plan

"""
import collections
import dataclasses
import pathlib
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple, TypeVar

import jsons
import typer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.types import StrPath

# Extracted from https://worksheets.codalab.org/worksheets/0x269ef752f8c344a28383240f7bb2be9c
domain_to_log_bundle = {
    "housing": "0x8923225a34c84d36a45328f9b9f9afe1",
    "calendar": "0xd1aec5cc914d4828afcf7a45a388a259",
    "socialnetwork": "0x2eaa3225265a4e0db25ae5cc17f235d9",
    "basketball": "0x36836301781d453dbdd0c92e26160742",
    "blocks": "0x3078fd8248b44e59b5019eaa6a2218f3",
    "publications": "0x3f0ad020cdb44d79ad6d764f3bc2959e",
    "recipes": "0x76ba3fd5373942b1b3eb959ede5da463",
    "restaurants": "0xfb9886deba0b43b9bb1387729427f96b",
}
data_bundle = "0x7cff4434e6ee412b94eb602e984d1d98"


# Output in .canonical.json
@dataclass(frozen=True, eq=True)
class CanonicalInfo:
    paraphrases: FrozenSet[str]
    formula: Optional[str]
    denotation: Optional[str]


# Output as .jsonl files
@dataclass
class ParaphraseExample:
    canonical: str
    natural: str
    formula: str


# The below are used in intermediate processing and not part of the output.
@dataclass(frozen=True, eq=True)
class DenotationExample:
    natural: str
    formula: str
    denotation: str


@dataclass(frozen=True, eq=True)
class ParaphraseGroup:
    canonical: str
    paraphrases: FrozenSet[str]


@dataclass(frozen=True, eq=True)
class PlanInfo:
    denotations: Set[str] = dataclasses.field(default_factory=set)
    formulas: Set[str] = dataclasses.field(default_factory=set)


def read_paraphrase_groups(path: StrPath) -> Iterator[ParaphraseGroup]:
    with open(path) as file:
        line_iter = iter(file)  # type: ignore
        current = None
        returned = None

        def inc() -> str:
            nonlocal current, returned
            if returned is None:
                current = next(line_iter)
            else:
                current = returned
                returned = None
            return current

        def put_back() -> None:
            nonlocal current, returned
            returned = current

        finished = False
        while not finished:
            try:
                line = inc()
            except StopIteration:
                return
            original_m = re.match("^original - (.*)$", line)
            assert original_m is not None, line
            original = original_m.group(1)

            paraphrases = []
            while True:
                try:
                    para_m = re.match("^  para - (.*?), worker - (.*)$", inc())
                    if para_m is None:
                        put_back()
                        break
                    paraphrases.append(para_m.group(1))
                except StopIteration:
                    finished = True
                    break
            yield ParaphraseGroup(
                canonical=original, paraphrases=frozenset(paraphrases)
            )


paraphrase_example_re = re.compile(
    r"\(example\s+\(utterance\s+\"(?P<utterance>.*)\"\)\s+\(original\s+\"(?P<original>.*)\"\)\s+\(targetFormula\s+(?P<formula>.*)"
)


def read_paraphrase_examples(path: StrPath) -> Iterator[ParaphraseExample]:
    with open(path) as file:
        contents = file.read()
        for m in paraphrase_example_re.finditer(contents):
            yield ParaphraseExample(
                natural=m.group("utterance"),
                canonical=m.group("original"),
                formula=m.group("formula"),
            )


denotation_re = re.compile(
    r"iter=1\.(?P<section>train|test): .*?(?P<index>\d+) \{\n *Example: (?P<natural>.*?) \{[^\}]*?targetFormula: (?P<formula>.*)\s+targetValue: (?P<denotation>.*)\n"
)


def find_denotations(
    log_path: StrPath,
) -> Tuple[List[DenotationExample], List[DenotationExample]]:
    train_dict = collections.defaultdict(set)
    test_dict = collections.defaultdict(set)

    with open(log_path) as f:
        log_contents = f.read()
        for m in denotation_re.finditer(log_contents):
            section = m.group("section")
            if section == "train":
                train_dict[int(m.group("index"))].add(
                    DenotationExample(
                        m.group("natural"), m.group("formula"), m.group("denotation")
                    )
                )
            elif section == "test":
                # test_dict[int(m.group("index"))].add(m.group("denotation"))
                test_dict[int(m.group("index"))].add(
                    DenotationExample(
                        m.group("natural"), m.group("formula"), m.group("denotation")
                    )
                )
            else:
                raise ValueError(section)

    train: List[DenotationExample] = []
    test: List[DenotationExample] = []
    for d, lst in ((train_dict, train), (test_dict, test)):
        for i in range(len(d)):  # pylint: disable=consider-using-enumerate
            assert i in d
            assert len(d[i]) == 1
            lst.append(next(iter(d[i])))

    return train, test


T = TypeVar("T")


def shuffle_and_partition(
    random_seed: int, lst: List[T], first_size: int
) -> Tuple[List[T], List[T]]:
    assert first_size < len(lst)
    lst = list(lst)
    rand = random.Random(random_seed)
    rand.shuffle(lst)
    return lst[:first_size], lst[first_size:]


def main(
    output_dir: pathlib.Path = typer.Option(
        pathlib.Path(__file__).resolve().parent / "data"
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir_str:
        tempdir = pathlib.Path(tempdir_str)

        # Download the main bundle
        subprocess.check_call(
            [
                "cl",
                "down",
                "-o",
                tempdir / "overnightData",
                f"https://worksheets.codalab.org::{data_bundle}",
            ]
        )

        for dataset_name, log_bundle_id in domain_to_log_bundle.items():
            log_path = tempdir / f"{dataset_name}.stdout"
            subprocess.check_call(
                [
                    "cl",
                    "down",
                    "-o",
                    log_path,
                    f"https://worksheets.codalab.org::{log_bundle_id}/stdout",
                ]
            )

            # Parse groups
            paraphrase_groups = list(
                read_paraphrase_groups(
                    tempdir / "overnightData" / f"{dataset_name}.paraphrases.groups"
                )
            )

            train_with_dev = list(
                read_paraphrase_examples(
                    tempdir
                    / "overnightData"
                    / f"{dataset_name}.paraphrases.train.examples"
                )
            )
            test = list(
                read_paraphrase_examples(
                    tempdir
                    / "overnightData"
                    / f"{dataset_name}.paraphrases.test.examples"
                )
            )
            train_with_dev_denotations, test_denotations = find_denotations(log_path)

            canonical_to_plan_info: Dict[str, PlanInfo] = collections.defaultdict(
                PlanInfo
            )
            for examples, denotations in (
                (train_with_dev, train_with_dev_denotations),
                (test, test_denotations),
            ):
                for para_example, denot_example in zip(examples, denotations):
                    assert para_example.natural == denot_example.natural
                    assert para_example.formula == denot_example.formula

                    plan_info = canonical_to_plan_info[para_example.canonical]
                    plan_info.denotations.add(denot_example.denotation)
                    plan_info.formulas.add(para_example.formula)

            print(
                f"{dataset_name}: {len(train_with_dev)} train, {len(test)} test ({100 * (len(test) / (len(test) + len(train_with_dev))):.2f}%), {len(paraphrase_groups)} canonical"
            )

            dev, train_without_dev = shuffle_and_partition(0, train_with_dev, len(test))
            for section_name, section in (
                ("train_with_dev", train_with_dev),
                ("train_without_dev", train_without_dev),
                ("dev", dev),
                ("test", test),
            ):
                with open(
                    output_dir / f"{dataset_name}.{section_name}.jsonl", "w"
                ) as f:
                    for item in section:
                        f.write(jsons.dumps(item))
                        f.write("\n")

            canonical_dict: Dict[str, CanonicalInfo] = {}
            for paraphrase_group in paraphrase_groups:
                assert paraphrase_group.canonical not in canonical_dict
                if paraphrase_group.canonical in canonical_to_plan_info:
                    plan_info = canonical_to_plan_info[paraphrase_group.canonical]
                    assert len(plan_info.denotations) == 1
                    assert len(plan_info.formulas) == 1
                    formula = next(iter(plan_info.formulas))
                    denotation = next(iter(plan_info.denotations))
                else:
                    print(
                        f"{dataset_name}: {paraphrase_group.canonical!r} missing denotation"
                    )
                    formula = None
                    denotation = None

                canonical_dict[paraphrase_group.canonical] = CanonicalInfo(
                    paraphrase_group.paraphrases, formula, denotation,
                )

            with open(output_dir / f"{dataset_name}.canonical.json", "w") as f:
                f.write(jsons.dumps(canonical_dict))


if __name__ == "__main__":
    typer.run(main)
