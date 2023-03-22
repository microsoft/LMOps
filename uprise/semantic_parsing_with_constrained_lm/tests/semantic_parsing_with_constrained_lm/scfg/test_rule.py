# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.rule import (
    Diff,
    all_possible_options,
    modify_expansion,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import NonterminalToken, TerminalToken
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import parse_string


@pytest.fixture(scope="module", name="parser")
def create_parser():
    return get_scfg_parser("start_for_test")


def test_all_possible_options(parser):

    assert all_possible_options(
        parse_string(parser, 'yes? "the"? choice "one"')
        == [
            (
                parse_string(parser, 'yes "the" choice "one"'),
                Diff(
                    included=[
                        TerminalToken("the", optional=True),
                        NonterminalToken("yes", optional=True),
                    ],
                    excluded=[],
                ),
            ),
            (
                parse_string(parser, 'yes choice "one"'),
                Diff(
                    included=[NonterminalToken("yes", optional=True)],
                    excluded=[TerminalToken("the", optional=True)],
                ),
            ),
            (
                parse_string(parser, '"the" choice "one"'),
                Diff(
                    included=[TerminalToken("the", optional=True)],
                    excluded=[NonterminalToken("yes", optional=True)],
                ),
            ),
            (
                parse_string(parser, 'choice "one"'),
                Diff(
                    included=[],
                    excluded=[
                        TerminalToken("the", optional=True),
                        NonterminalToken("yes", optional=True),
                    ],
                ),
            ),
        ]
    )


def test_modify_expansion(parser):
    assert modify_expansion(
        parse_string(parser, '"hi" x? "my name is" y? z?'),
        Diff(
            [NonterminalToken("x", optional=True)],
            [NonterminalToken("z", optional=True)],
        ),
    ) == tuple(parse_string(parser, '"hi" x "my name is" y?'))
