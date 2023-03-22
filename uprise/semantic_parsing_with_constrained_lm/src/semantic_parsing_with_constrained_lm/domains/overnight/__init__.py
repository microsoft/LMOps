# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from transformers import PreTrainedTokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.trie import Trie
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.eval import TopKExactMatch
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.trie_partial_parse import TriePartialParse


class OutputType(str, Enum):
    Utterance = "utterance"
    MeaningRepresentation = "meaningRepresentation"


@dataclass
class TopKDenotationMatch(TopKExactMatch[FullDatum]):
    canonical_to_denotation: Dict[str, str]

    def _is_correct(self, pred: str, datum: FullDatum) -> bool:
        target = datum.canonical
        pred_denotation = self.canonical_to_denotation.get(pred)
        target_denotation = self.canonical_to_denotation.get(target, None)
        if pred_denotation is None and target_denotation is None:
            return pred == target
        else:
            return pred_denotation == target_denotation


@dataclass
class OvernightPieces:
    train_data: List[FullDatum]
    test_data: List[FullDatum]
    partial_parse_builder: Callable[[Datum], TriePartialParse]
    denotation_metric: TopKDenotationMatch

    @staticmethod
    def from_dir(
        tokenizer: PreTrainedTokenizer,
        root_dir: pathlib.Path,
        domain: str,
        is_dev: bool,
        k: int,
        output_type: OutputType = OutputType.Utterance,
        simplify_logical_forms=False,
        prefix_with_space=False,
    ) -> "OvernightPieces":
        # TODO make this configurable?
        canonical_data = json.load(open(root_dir / f"{domain}.canonical.json"))

        if output_type == OutputType.Utterance:
            target_output_to_denotation = {
                k: v["denotation"] for k, v in canonical_data.items()
            }
            datum_key = "canonical"
        elif output_type == OutputType.MeaningRepresentation:
            target_output_to_denotation = {}
            for program_info in canonical_data.values():
                formula = program_info["formula"]
                if formula is None:
                    continue
                if simplify_logical_forms:
                    formula = OvernightPieces.simplify_lf(formula)
                assert formula not in target_output_to_denotation
                target_output_to_denotation[formula] = program_info["denotation"]
            datum_key = "formula"
        else:
            raise ValueError(output_type)

        train_data, test_data = [
            [
                FullDatum(
                    dialogue_id=f"{dataset_name}-{i}",
                    turn_part_index=None,
                    agent_context=None,
                    natural=d["natural"],
                    canonical=OvernightPieces.simplify_lf(d[datum_key])
                    if simplify_logical_forms
                    else d[datum_key],
                )
                for i, line in enumerate(open(path))
                for d in [json.loads(line)]
            ]
            for dataset_name, path in (
                (
                    "train",
                    root_dir
                    / f"{domain}.train_with{'out' if is_dev else ''}_dev.jsonl",
                ),
                ("eval", root_dir / f"{domain}.{'dev' if is_dev else 'test'}.jsonl",),
            )
        ]
        if prefix_with_space:
            canonical_trie = Trie(
                tokenizer.encode(" " + canon) for canon in target_output_to_denotation
            )
        else:
            canonical_trie = Trie(
                tokenizer.encode(canon) for canon in target_output_to_denotation
            )
        partial_parse_builder = lambda _: TriePartialParse(canonical_trie)

        return OvernightPieces(
            train_data,
            test_data,
            partial_parse_builder,
            TopKDenotationMatch(k, target_output_to_denotation),
        )

    @staticmethod
    def simplify_lf(lf: str) -> str:
        return lf.replace("edu.stanford.nlp.sempre.overnight.SimpleWorld.", "")
