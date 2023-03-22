# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC
from dataclasses import replace
from typing import List, Optional, Tuple

from pydantic.dataclasses import dataclass

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import EntitySchema, OptionableSCFGToken
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion


@dataclass(frozen=True)
class Rule(ABC):
    lhs: str


@dataclass(frozen=True)
class PlanRule(Rule):
    lhs: str
    rhs: Expansion


@dataclass(frozen=True)
class SyncRule(Rule):
    lhs: str
    utterance_rhss: List[Expansion]
    plan_rhs: Expansion
    entity_schema: Optional[EntitySchema]


@dataclass(frozen=True)
class UtteranceRule(Rule):
    lhs: str
    utterance_rhss: List[Expansion]


@dataclass(frozen=True)
class Diff:
    included: Tuple[OptionableSCFGToken, ...]
    excluded: Tuple[OptionableSCFGToken, ...]


def all_possible_options(tokens: Expansion) -> List[Tuple[Expansion, Diff]]:
    """
    Given an expansion with optional tokens like: yes? "the"? choice "one"
    Generate all implied possible tokenized strings with no optional tokens, and for each possibility,
    return a list of: (the generated expansion,
                       Diff(a list of optional tokens that were included,
                       a list of the optional tokens that were left out,
                       ))

    For example, given [yes?, "the"?, choice, "one"] return
        (["yes", '"the"', "choice", '"one"'], Diff([], ['"the"?', "yes?"])),
        (["yes", "choice", '"one"'], Diff(['"the"?'], ["yes?"])),
        (['"the"', "choice", '"one"'], Diff(["yes?"], ['"the"?'])),
        (["choice", '"one"'], Diff(['"the"?', "yes?"], [])),
    """
    if not tokens:
        return [((), Diff((), ()))]

    first_token = tokens[0]
    all_suffixes = all_possible_options(tokens[1:])

    if isinstance(first_token, OptionableSCFGToken) and first_token.optional:
        include = [
            (
                (replace(first_token, optional=False),) + suffix,
                replace(diff, included=diff.included + (first_token,)),
            )
            for suffix, diff in all_suffixes
        ]

        exclude = [
            (suffix, replace(diff, excluded=diff.excluded + (first_token,)))
            for suffix, diff in all_suffixes
        ]

        return include + exclude
    else:
        return [((first_token,) + suffix, diff) for suffix, diff in all_suffixes]


def modify_expansion(tokens: Expansion, diff: Diff) -> Expansion:
    """
    Modify tokens according to the diff: keep and strip regex from the ones in diff.included,
    and remove the ones in diff.excluded.
    For example,
    match_tokens(['"hi"', 'x?', 'my name is', 'y?', 'z?'], Diff(['x?'], ['z?'])) returns
    ['"hi"', 'x', 'my name is', 'y?']

    Note that we kept the 'y?' token unchanged because it didn't appear in the diff.
    """
    new_expansion = [
        replace(token, optional=False) if token in diff.included else token
        for token in tokens
        if token not in diff.excluded
    ]

    return tuple(new_expansion)
