# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Methods that illustrates the simplest ways to use EarleyChart:
as a recognizer of a token string, or as a generator of grammatical sentences.
"""

from typing import Iterable, Iterator, List, cast

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.earley import EarleyLRChart, PackedForest
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import Grammar, Terminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.input import ListPosition, SigmaStarTriePosition


def parse(
    sentence: Iterable[Terminal], grammar: Grammar[Terminal]
) -> PackedForest[Terminal]:
    start_pos = ListPosition(list(sentence))
    chart = EarleyLRChart(grammar=grammar, start_pos=start_pos)
    return chart.parse()


def is_grammatical(tokens: List[str], grammar: Grammar[str]) -> bool:
    """
    Tests whether the given input `tokens` are grammatical under `grammar`.
    """
    start_pos = ListPosition(tokens)
    chart = EarleyLRChart(grammar, start_pos)
    for _ in chart.accepting_positions():
        return True  # we're grammatical if the iterator is non-empty
    return False


def enumerate_sentences(grammar: Grammar[str]) -> Iterator[List[str]]:
    """
    Yields grammatical sentences in length order (may not terminate).
    """
    # root of a Σ* trie with string-labeled edges (as the grammar uses Terminal=str)
    start_pos = SigmaStarTriePosition[str]()
    chart = EarleyLRChart(grammar, start_pos)
    for pos in chart.accepting_positions():  # enumerate nodes in the Σ* trie
        # necessary because current typing isn't strong enough
        _pos = cast(SigmaStarTriePosition[str], pos)
        yield _pos.prefix()
