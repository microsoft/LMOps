# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from random import shuffle
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

from lark import Token, Tree
from more_itertools import roundrobin

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.agenda import BP, Attach, Meta, Predict, Scan
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.earley import EarleyLRChart, PackedForest
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import DottedRule, Grammar, Nonterm, Symbol
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.input import Position
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.util import IteratorGenerator
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import (
    NonterminalToken,
    RegexToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Alias, Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import GrammarRules

# a str of length 1. not enforced statically, but that's what it should be
Char = str
# our Terminals are either a char or a regex.
# we assume regexes also run on single chars.
CharTerminal = Union[Char, RegexToken]

START: Nonterm = Nonterm("start")


@dataclass(frozen=True)
class CharPosition(Position[CharTerminal]):
    """
    A position in an untokenized string of characters.
    Allows us to tokenize as we go.
    """

    string: str
    i: int = 0

    def scan(self, terminal: CharTerminal) -> Iterable["CharPosition"]:
        if isinstance(terminal, RegexToken):
            next_char = self.string[self.i : self.i + 1]
            if terminal.compiled.match(string=next_char):
                return [CharPosition(self.string, self.i + 1)]
        else:
            if self.string.startswith(terminal, self.i):
                return [CharPosition(self.string, self.i + len(terminal))]
        return []

    def is_final(self) -> bool:
        return self.i == len(self.string)

    def __len__(self) -> int:
        return self.i

    def __repr__(self) -> str:
        return self.string[: self.i] + "^" + self.string[self.i :]


@dataclass(frozen=True)
class CharGrammar(Grammar[CharTerminal]):
    """
    Currently our Earley/GPT integration requires a character-level grammar
    (really a byte-level grammar) to handle the mismatch between grammar
    tokenization and GPT tokenization.
    In the future, we plan on relaxing this requirement by letting
    `Position`s and `DottedRule`s represent partially consumed grammar
    terminals.
    """

    def parse_forest(self, sentence: str) -> PackedForest[CharTerminal]:
        """Parses to a possibly ambiguous PackedForest"""
        start_pos = CharPosition(sentence)
        chart = EarleyLRChart(self, start_pos)
        return chart.parse()

    def parses(self, sentence: str, max_depth: Optional[int] = None) -> Iterator[Tree]:
        """Parses to an iterator of unambiguous Trees"""
        if max_depth is None:
            max_depth = len(sentence)
        start_pos = CharPosition(sentence)
        chart = EarleyLRChart(self, start_pos)
        meta = chart.final_meta()
        return _generate_from_meta(meta=meta, max_depth=max_depth)

    @staticmethod
    def from_aliased_grammar(
        grammar: Dict[Nonterminal, List[Tuple[Expansion, Alias]]]
    ) -> "CharGrammar":
        def convert_rhs(expansion: Expansion) -> Tuple[Symbol[CharTerminal], ...]:
            result = []
            for token in expansion:
                if isinstance(token, TerminalToken):
                    for char in token.render():
                        result.append(char)
                elif isinstance(token, RegexToken):
                    result.append(token)
                elif isinstance(token, NonterminalToken):
                    result.append(Nonterm(token.value))
            return tuple(result)

        rules: Dict[Nonterm, List[DottedRule[CharTerminal]]] = {
            Nonterm(origin): [
                DottedRule.from_rule(Nonterm(origin), convert_rhs(rhs), alias=alias)
                for rhs, alias in rhss
            ]
            for origin, rhss in sorted(grammar.items())
        }
        return CharGrammar(root=START, expansions=rules)  # type: ignore

    @staticmethod
    def from_preprocessed_rules(rules: GrammarRules):
        aliased_grammar = {
            lhs: [(rhs, None) for rhs in rhss] for lhs, rhss in rules.items()
        }
        return CharGrammar.from_aliased_grammar(aliased_grammar)  # type: ignore


def _generate_from_meta(meta: Meta[CharTerminal], max_depth: int) -> Iterator[Tree]:
    # TODO: This has the same structure as `backtrace`. refactor
    def go(m: Meta[CharTerminal], d: int) -> Iterator[Tree]:
        if d > 0:
            # TODO: sort by progress, put unaries, and especially self-loops at the end
            alternatives = list(m.bps)
            # don't infinite loop unless there is no other alternative
            # TODO: is this good?
            shuffle(alternatives)
            yield from roundrobin(*[go_bp(b, d) for b in alternatives])

    def go_bp(bp: BP, d: int) -> Iterator[Tree]:
        """Backtrace a single backpointer"""
        last_child_trees: Iterable[Optional[Tree]]
        # Beginning of a rule (includes empty productions)
        if isinstance(bp, Predict):
            alias = bp.new_item.dotted_rule.alias
            yield Tree(data=alias, children=[])
        # a completed NT
        elif isinstance(bp, Attach):
            # the parent production, e.g. `W -> X Y . Z`
            parent_item = bp.customer
            # the right-most child of parent, e.g. `Z`
            last_child_item = bp.server
            last_child_meta = bp.col.get_meta(last_child_item)
            # TODO: hack specific to auto_grammar
            if "nonquote" in parent_item.dotted_rule.lhs.name:
                new_depth = d
            else:
                new_depth = d - 1
            # generate from the right-most child
            last_child_trees = IteratorGenerator(lambda: go(last_child_meta, new_depth))
            parent_meta = last_child_item.start_col.get_meta(parent_item)
            # backtrace through the rest of the children (e.g. `X Y`)
            yield from go_init(parent_meta, last_child_trees, d)
        # a T
        elif isinstance(bp, Scan):
            # the parent production, e.g. `W -> X Y . /z/`
            parent_item = bp.item
            if isinstance(bp.terminal, RegexToken):
                # look up the concrete char in the input string
                pos = cast(CharPosition, bp.col.pos)
                ith_char = pos.string[pos.i : pos.i + 1]
                last_child_trees = [Token("", ith_char)]  # type: ignore
            else:
                # non-regex Leaves are ignored in lark.Trees
                last_child_trees = [None]
            parent_meta = bp.col.get_meta(parent_item)
            # backtrace through the rest of the children (e.g. `X Y`)
            yield from go_init(parent_meta, last_child_trees, d)
        yield from ()  # unreachable

    def go_init(
        parent_meta: Meta[CharTerminal],
        last_child_trees: Iterable[Optional[Tree]],
        d: int,
    ) -> Iterator[Tree]:
        """Backtrace through parent, then append last_child_node to the list of children"""
        for maybe_last_child_node in last_child_trees:
            for init_node in go(parent_meta, d):
                # stitch it back together with the right-most child
                last_children: List[Union[str, Tree]] = (
                    [maybe_last_child_node] if maybe_last_child_node is not None else []
                )
                children = init_node.children + last_children
                yield Tree(data=init_node.data, children=children)

    return go(meta, max_depth)
