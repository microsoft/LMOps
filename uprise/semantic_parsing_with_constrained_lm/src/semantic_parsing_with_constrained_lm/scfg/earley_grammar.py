# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Tuple, Union

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import DottedRule, Grammar, Nonterm, Symbol
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.char_grammar import START
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import (
    EmptyToken,
    NonterminalToken,
    RegexToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Alias, Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import GrammarRules

CFTerminal = Union[str, RegexToken]


class EarleyCFGrammar(Grammar[CFTerminal]):
    """A grammar for one of the two sides of an SCFG.

    Similar to CharGrammar, but it doesn't split up all terminals into single
    characters.
    """

    @staticmethod
    def from_preprocessed_rules(rules: GrammarRules):
        aliased_grammar = {
            lhs: [(rhs, None) for rhs in rhss] for lhs, rhss in rules.items()
        }
        return EarleyCFGrammar.from_aliased_grammar(aliased_grammar)  # type: ignore

    @staticmethod
    def from_aliased_grammar(
        grammar: Dict[Nonterminal, List[Tuple[Expansion, Alias]]]
    ) -> "EarleyCFGrammar":
        def convert(expansion: Expansion) -> Tuple[Symbol[CFTerminal], ...]:
            result = []
            for token in expansion:
                if isinstance(token, TerminalToken):
                    result.append(token.render())
                elif isinstance(token, RegexToken):
                    result.append(token)
                elif isinstance(token, NonterminalToken):
                    result.append(Nonterm(token.value))
                elif isinstance(token, EmptyToken):
                    pass
                else:
                    raise ValueError(token)
            return tuple(result)

        return EarleyCFGrammar(
            root=START,
            expansions={
                Nonterm(origin): [
                    DottedRule.from_rule(Nonterm(origin), rhs=convert(rhs), alias=alias)
                    for rhs, alias in rhss
                ]
                for origin, rhss in sorted(grammar.items())
            },
        )
