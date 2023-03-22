# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC
from collections.abc import Sequence, Sized
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

# This is a type variable, since the terminal symbols of the CFG could
# be anything that matches against input prefixes (e.g., regexps or integerized tokens).
# Make sure that the Terminal type is disjoint from Nonterm, since any Nonterm
# on a rule RHS will be treated as a Nonterm.
Terminal = TypeVar("Terminal")

# Elements of a rule RHS.
Symbol = Union["Nonterm", Terminal]


@dataclass(frozen=True)
class Nonterm:
    """Grammar nonterminal"""

    name: str

    def __repr__(self):
        return f"Nonterm({self.name})"

    def __str__(self):
        return self.name

    # TODO: consider interning the string or using an integerizer, as a minor speedup
    # NOTE: nonterminals are sometimes structured (attributes, slashed gaps, etc.), so
    #       really we should make an abstract base class, and parameterize grammars
    #       on the particular Nonterm type (using a bounded type variable so it must
    #       be a subclass of that base class).


# TODO: extract abstract superclass, implement FSADottedRule(AbstractDottedRule)
#  to directly allow rules whose right-hand side is a (possibly minimal)
#  regexp over terminals or nonterminals.
@dataclass(frozen=True, eq=True, unsafe_hash=True)
class DottedRule(Generic[Terminal]):
    """
    The residue of a grammar rule after part of it has been matched.
    A complete grammar rule (with the dot at the start) is a special case.
    A basic dotted rule, whose RHS consists of a sequence of remaining symbols
    that we need to match in order to complete the rule.
    We don't bother to store the "pre-dot" symbols that have already been
    matched.  That leads to a speedup in Earley's algorithm through more
    aggressive consolidation of duplicates.
    """

    # the left-hand side of the rule.  The other methods match against the right-hand side.
    lhs: Nonterm
    # TODO: we don't need this for recognition, only parsing. This prevents us from
    # recombining Items that share a common unconsumed suffix.
    # Could even recognize first, and then rerun with backpointers?
    # The full original rhs of the rule
    full_rhs: Tuple[Symbol[Terminal], ...]
    # TODO: turn this into a state in a reverse trie of rules, for hashing efficiency
    # The unconsumed portion of the rhs
    rhs: Tuple[Symbol[Terminal], ...]
    # a distinct identifier for this rule
    alias: Optional[str] = None

    @staticmethod
    def from_rule(
        lhs: Nonterm, rhs: Tuple[Symbol[Terminal], ...], alias: Optional[str] = None
    ) -> "DottedRule[Terminal]":
        return DottedRule(lhs, rhs, rhs, alias=alias)

    def next_symbols(self) -> Iterable[Symbol[Terminal]]:
        return self.rhs[:1]

    def is_final(self) -> bool:
        """Dot is past the last NT"""
        return not self.rhs  # empty tuple

    def is_initial(self) -> bool:
        """Dot is before the first NT"""
        return len(self.full_rhs) == len(self.rhs) and self.full_rhs[0] == self.rhs[0]

    def scan(self, symbol: Symbol[Terminal]) -> Iterable["DottedRule[Terminal]"]:
        """
        Returns new DottedRules where the symbol has been consumed if it matches the RHS.

        If the symbol exactly matches the first element of `rhs`, then it is removed from `rhs`.
        If the symbol is a prefix of the first element of `rhs`, then the prefix is removed.

        Example `rhs`: ("meeting", "at", Nonterminal(time))
        If symbol is "meeting", then the new `rhs` is ("at", Nonterminal(time)).
        If symbol is "meet", then the new `rhs` is ("ing", "at", Nonterminal(time)).
        """

        if not self.rhs:
            return ()
        first_rhs = self.rhs[0]
        if first_rhs == symbol:
            # Exact match case: drop first element of RHS
            return (
                DottedRule[Terminal](
                    lhs=self.lhs,
                    full_rhs=self.full_rhs,
                    rhs=self.rhs[1:],
                    alias=self.alias,
                ),
            )
        if isinstance(first_rhs, Nonterm) or isinstance(symbol, Nonterm):
            # If either is Nonterm, then a partial match is impossible, so give up here.
            return ()

        if (
            isinstance(first_rhs, Sequence)
            and isinstance(symbol, Sized)
            and first_rhs[: len(symbol)] == symbol
        ):
            # Partial match succeeded.
            return (
                DottedRule[Terminal](
                    lhs=self.lhs,
                    full_rhs=self.full_rhs,
                    rhs=(first_rhs[len(symbol) :],) + self.rhs[1:],
                    alias=self.alias,
                ),
            )
        return ()

    def __repr__(self) -> str:
        def to_str(xs):
            return " ".join(str(sym) for sym in xs)

        consumed_len = len(self.full_rhs) - len(self.rhs)
        consumed = self.full_rhs[:consumed_len]
        return f'DottedRule("{self.lhs} → {to_str(consumed)} ⋯ {to_str(self.rhs)}")'


@dataclass(frozen=True)
class Grammar(Generic[Terminal], ABC):
    """A context-free grammar."""

    root: Nonterm
    expansions: Dict[Nonterm, List[DottedRule[Terminal]]]
