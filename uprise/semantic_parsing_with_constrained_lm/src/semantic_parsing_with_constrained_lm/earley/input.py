# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Implement a position in a language model trie whose edges are labeled
# with word pieces (so, really a radix tree; we can cache the outgoing edges
# from each node rather than repeatedly querying the language model).
#
# Scanning a terminal might traverse multiple edges because the terminal
# consists of several word pieces, or part of an edge, because a terminal
# matches only part of a word piece (in which case the returned position will be
# partway along an edge, which we can represent as the destination position of
# that edge paired with a residual substring that we still have to read to
# arrive at that destination position).  Both can happen at once, e.g., a
# terminal might complete an existing partial edge and then traverse 2 more
# edges and then go partway along the next edge, like staggered bricks in a
# wall.
#
# The design for this is discussed at
# https://semanticmachines.slack.com/archives/C01BZ09JLHL/p1611830809374000?thread_ts=1611773531.333600&cid=C01BZ09JLHL
# https://semanticmachines.slack.com/archives/C01BZ09JLHL/p1611866365384800?thread_ts=1611773531.333600&cid=C01BZ09JLHL
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, Sized, Tuple, cast

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import Terminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.keydefaultdict import KeyDefaultDict


# pylint: disable=inherit-non-class
class Position(ABC, Sized, Generic[Terminal]):
    """
    An abstract representation of a position in an input string, lattice, or trie.
    Concrete implementations of Position might also provide access to the prefix read so far
    and the weight of that prefix.
    """

    @abstractmethod
    def scan(self, terminal: Terminal) -> Iterable["Position[Terminal]"]:
        """
        Return all the new positions we can get to from here by scanning terminal.

        TODO: The type on this is not precise enough.  We really want (P, Terminal) -> Iterable[P],
        and we should be able to write self: P to do this, but we can't declare P to be bounded
        by Position[Terminal] because that's generic.
        """
        pass

    @abstractmethod
    def is_final(self) -> bool:
        """Is this a final position in the input, i.e., are we allowed to end here?"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        A measurement of how far along the position is in the input (e.g., the prefix length).
        May be used by some control strategies to process positions in left-to-right order.
        """
        pass

    def __lt__(self, other: "Position[Terminal]") -> bool:
        return len(self) < len(other)


class ListPosition(Position[Terminal]):
    """A position in a list of tokens."""

    def __init__(self, tokens: List[Terminal], i: int = 0):
        self._tokens: List[Terminal] = tokens
        self._i: int = i

    def scan(self, terminal: Terminal) -> Iterable["ListPosition[Terminal]"]:
        if self.is_final() or self._tokens[self._i] != terminal:
            return ()
        else:
            return (ListPosition[Terminal](self._tokens, self._i + 1),)

    def is_final(self) -> bool:
        return self._i == len(self._tokens)

    def __len__(self) -> int:
        return self._i

    def __repr__(self) -> str:
        return str(self._i)


@dataclass(frozen=True)
class StringPosition(Position[str]):
    """
    A position in an untokenized string of characters.
    Allows us to tokenize as we go.
    """

    string: str
    i: int = 0

    def scan(self, terminal: str) -> Iterable["StringPosition"]:
        if self.string.startswith(terminal, self.i):
            return [StringPosition(self.string, self.i + len(terminal))]
        else:
            return []

    def is_final(self) -> bool:
        return self.i == len(self.string)

    def __len__(self) -> int:
        return self.i

    def __repr__(self) -> str:
        return self.string[: self.i] + "^" + self.string[self.i :]


class SigmaStarTriePosition(Position[Terminal]):
    """
    A Position in a lazy trie of all possible token sequences (Σ*).
    Thus, it matches any prefix and retains the history of that prefix.
    This is useful in enumerating all strings that are accepted by the grammar.
    """

    # parent node if any
    _prev: Optional["SigmaStarTriePosition[Terminal]"]
    # label of edge to us from parent
    _last: Optional[Terminal]
    # length of path to us from root
    _len: int
    # transition function to child node: expanded on demand
    _next: KeyDefaultDict[Terminal, "SigmaStarTriePosition[Terminal]"]

    def __init__(
        self, edge: Optional[Tuple["SigmaStarTriePosition[Terminal]", Terminal]] = None
    ):
        """Constructs the root trie node by default, or a child of another node via a labeled edge."""
        if edge is None:
            self._prev, self._last, self._len = None, None, 0  # root node
        else:
            self._prev, self._last = edge  # child node
            self._len: int = 0 if self._prev is None else 1 + len(self._prev)

        self._next = KeyDefaultDict(
            lambda terminal: SigmaStarTriePosition[Terminal]((self, terminal))
        )

    def is_final(self) -> bool:
        return True  # any string is a valid prefix, so we can always stop here

    def scan(self, terminal: Terminal) -> Iterable["SigmaStarTriePosition[Terminal]"]:
        return (self._next[terminal],)

    def __len__(self) -> int:
        return self._len

    def last(self) -> Optional[Terminal]:
        """The Terminal immediately before this Position, if any."""
        return self._last

    def prefix(self) -> List[Terminal]:
        """The sequence of Terminals leading up to this Position."""
        if self._prev is None:
            return []
        else:
            result = self._prev.prefix()
            result.append(cast(Terminal, self._last))
            return result

    def __repr__(self) -> str:
        return repr(self.prefix())
