# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import json
import signal
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from operator import itemgetter
from typing import Dict, Iterable, List, Optional, Set

from dataflow.core.lispress import (
    lispress_to_program,
    parse_lispress,
    program_to_lispress,
    render_compact,
    try_round_trip,
)
from lark import GrammarError, ParseError, UnexpectedCharacters

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.generate import parse_and_render
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import FullDatum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow.disambiguate import score_auto_grammar_plan
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.eval import TopKExactMatch


class CalflowOutputLanguage(str, Enum):
    Canonical = "canonicalUtterance"
    Lispress = "lispress"


@dataclass(frozen=True)
class CalflowDatum(FullDatum):
    lispress: str


def read_calflow_jsonl(
    filename: StrPath,
    model_output_type: CalflowOutputLanguage,
    whitelisted_dialogue_ids: Optional[Set[str]] = None,
) -> List[CalflowDatum]:
    """
    Reads CalflowDatum lists from file `filename` with `canonical` based on model_output_type. Selects based on
    whitelisted_dialogue_ids when set, reads all data otherwise.
    """
    with open(filename) as test_file:
        return [
            CalflowDatum(
                agent_context=json_line.get("context", ""),
                natural=json_line["utterance"],
                canonical=json_line[model_output_type],
                dialogue_id=json_line["dialogueId"],
                turn_part_index=json_line["turnIndex"],
                lispress=json_line["lispress"],
            )
            for line in test_file
            for json_line in [json.loads(line)]
            if whitelisted_dialogue_ids is None
            or json_line["dialogueId"] in whitelisted_dialogue_ids
        ]


def predict_plan_from_canonical(
    scfg: SCFG, utterance: str, k: int = 1000, max_depth: int = 15,
) -> str:
    """
    Predicts a single Lispress surface string from the given canonical
    `utterance`.
    Finds possible parses using `scfg`, truncates to the top `k`, then picks
    the highest scoring possible parse under `score_auto_grammar_plan`.
    """
    unscored_plans: Iterable[str]
    try:
        unscored_plans = parse_and_render(
            scfg, utterance, source_is_plan=False, max_depth=max_depth
        )
    except (GrammarError, UnexpectedCharacters, ParseError, AttributeError):
        unscored_plans = []
    try:
        scored_plans = (
            (plan, score_auto_grammar_plan(plan)) for plan in unscored_plans
        )
        filtered_plans = (
            (plan, score) for plan, score in scored_plans if score != -float("inf")
        )
        truncated_plans = islice(filtered_plans, k)
    except AttributeError:
        truncated_plans = iter([])
    try:
        best_plan, _best_score = max(truncated_plans, key=itemgetter(1))
    except ValueError:
        # no candidates
        best_plan = " (FenceScope)"

    # Remove leading space
    # TODO: Remove need for this by removing the space from the grammar
    assert len(best_plan) > 0 and best_plan[0] == " "
    best_plan = best_plan[1:]

    return best_plan


class ParseTimeout(Exception):
    pass


@dataclass
class CalflowMetrics(TopKExactMatch[CalflowDatum]):
    scfg: SCFG
    data_type: CalflowOutputLanguage

    cached_parses: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Only attempt to convert predictions with the same length as the gold,
    # which saves a lot of time with parsing.
    require_exact_length: bool = False

    @staticmethod
    def parse_timeout_handler(sig, frame):
        raise ParseTimeout

    def cached_parse(self, pred: str, gold: Optional[str]) -> str:
        """
        Given a canonical utterance, convert it into a lispress plan.
        """
        if pred in self.cached_parses:
            return self.cached_parses[pred]
        if self.require_exact_length and (gold is None or len(pred) != len(gold)):
            return "(FenceScope)"

        prev_signal = signal.getsignal(signal.SIGALRM)
        if prev_signal == signal.SIG_DFL:
            signal.signal(signal.SIGALRM, self.parse_timeout_handler)
            signal.alarm(300)

        try:
            predicted_plan = predict_plan_from_canonical(self.scfg, " " + pred)
            signal.signal(signal.SIGALRM, prev_signal)
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            predicted_plan = "(FenceScope)"
        finally:
            signal.signal(signal.SIGALRM, prev_signal)

        self.cached_parses[pred] = predicted_plan
        return predicted_plan

    def _is_correct(self, pred: str, target: CalflowDatum) -> bool:
        if self.data_type == CalflowOutputLanguage.Canonical:
            predicted_plan = self.cached_parse(pred, target.canonical)
        else:
            try:
                # round-trip to canonicalize
                predicted_plan = render_compact(
                    program_to_lispress(lispress_to_program(parse_lispress(pred), 0)[0])
                )
            except Exception:  # pylint: disable=W0703
                predicted_plan = "(FenceScope)"

        return try_round_trip(target.lispress) == try_round_trip(predicted_plan)
