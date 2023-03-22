# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import DottedRule, Nonterm
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.earley_grammar import EarleyCFGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import RegexToken
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar

GRAMMAR_1 = """
start -> !"abc", "abc"
start -> !"a" b, "a" b
b -> !"bd", "bd"
""".strip()

GRAMMAR_2 = r"""
start -> quoted, quoted
quoted -> !"\"" nonquoteplus !"\"", "\"" nonquoteplus "\""
nonquoteplus -> !/[^"]/ nonquotestar, /[^"]/ nonquotestar
nonquotestar -> !/[^"]/ nonquotestar, /[^"]/ nonquotestar
nonquotestar -> #e, #e
""".strip()


def get_grammar(grammar_str: str) -> Tuple[EarleyCFGrammar, EarleyCFGrammar]:
    preprocessed_grammar = PreprocessedGrammar.from_line_iter(grammar_str.splitlines())
    utt_grammar = EarleyCFGrammar.from_preprocessed_rules(
        preprocessed_grammar.all_utterance_rules
    )
    plan_grammar = EarleyCFGrammar.from_preprocessed_rules(
        preprocessed_grammar.all_plan_rules
    )
    return utt_grammar, plan_grammar


def test_grammar_1():
    utt_grammar, plan_grammar = get_grammar(GRAMMAR_1)
    assert utt_grammar == plan_grammar
    assert utt_grammar.expansions == {
        Nonterm("start"): [
            DottedRule.from_rule(Nonterm("start"), ("abc",)),
            DottedRule.from_rule(Nonterm("start"), ("a", Nonterm("b"))),
        ],
        Nonterm("b"): [DottedRule.from_rule(Nonterm("b"), ("bd",)),],
    }


def test_grammar_2():
    utt_grammar, plan_grammar = get_grammar(GRAMMAR_2)
    assert utt_grammar == plan_grammar
    assert utt_grammar.expansions == {
        Nonterm("start"): [
            DottedRule.from_rule(Nonterm("start"), (Nonterm("quoted"),)),
        ],
        Nonterm("quoted"): [
            DottedRule.from_rule(
                Nonterm("quoted"), ('"', Nonterm("nonquoteplus"), '"')
            ),
        ],
        Nonterm("nonquoteplus"): [
            DottedRule.from_rule(
                Nonterm("nonquoteplus"),
                (
                    RegexToken('/[^"]/', optional=False, prefix=""),
                    Nonterm("nonquotestar"),
                ),
            ),
        ],
        Nonterm("nonquotestar"): [
            DottedRule.from_rule(
                Nonterm("nonquotestar"),
                (
                    RegexToken('/[^"]/', optional=False, prefix=""),
                    Nonterm("nonquotestar"),
                ),
            ),
            DottedRule.from_rule(Nonterm("nonquotestar"), ()),
        ],
    }
