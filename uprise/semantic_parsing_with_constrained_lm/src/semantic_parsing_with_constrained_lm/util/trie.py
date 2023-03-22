# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.missing_sentinel import MISSING_SENTINEL, MissingSentinel

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class TrieSetNode(Generic[K]):
    __slots__ = ("is_terminal", "children")
    # TODO: In the future, we can turn Trie into a mapping rather than a set
    # by storing arbitrary values here.
    is_terminal: bool
    children: Dict[K, "TrieSetNode[K]"]


@dataclass
class TrieMapNode(Generic[K, V]):
    __slots__ = ("value", "children")
    children: Dict[K, "TrieMapNode[K, V]"]
    value: Union[MissingSentinel, V]

    def find(self, key: Sequence[K]) -> "TrieMapNode[K, V]":
        node = self
        for elem in key:
            child_node = node.children.get(elem)
            if child_node is None:
                return None
            node = child_node
        return node

    def add(self, key: Sequence[K]) -> "TrieMapNode[K, V]":
        node = self
        for elem in key:
            child_node = node.children.get(elem)
            if child_node is None:
                child_node = node.children[elem] = TrieMapNode[K, V](
                    value=MISSING_SENTINEL, children={}
                )
            node = child_node
        return node

    @property
    def is_terminal(self) -> bool:
        # Unfortunately, even if you condition on this to access the node's value,
        # mypy will complain that it could be a MissingSentinel.
        return not isinstance(self.value, MissingSentinel)


@dataclass
class _SearchState(Generic[K]):
    __slots__ = ("node", "prefix")

    node: TrieSetNode
    prefix: Tuple[K, ...]


class Trie(MutableSet[Sequence[K]]):
    def __init__(self, iterable: Iterable[Sequence[K]] = ()):
        # Intended to be a recurisve type, but mypy doesn't support that
        self.root = TrieSetNode[K](is_terminal=False, children={})
        for elem in iterable:
            self.add(elem)

    def __contains__(self, seq: Any) -> bool:
        if not isinstance(  # pylint: disable=isinstance-second-argument-not-valid-type
            seq, Sequence
        ):
            return False
        node = self._prefix_to_node(seq)
        if node is None:
            return False
        return node.is_terminal

    def __iter__(self) -> Iterator[Sequence[K]]:
        """Returned elements are always tuples, even if other kinds of Sequences were added."""
        return self._iter_from_node(self.root)

    def __len__(self) -> int:
        return self._count_from_node(self.root)

    def add(self, seq: Sequence[K]) -> None:
        node = self.root
        for elem in seq:
            child_node = node.children.get(elem)
            if child_node is None:
                child_node = node.children[elem] = TrieSetNode[K](
                    is_terminal=False, children={}
                )
            node = child_node
        node.is_terminal = True

    def discard(self, seq: Sequence[K]) -> None:
        raise NotImplementedError

    # Below are methods not in the interface of MutableSet
    def prefix_next(self, prefix: Sequence[K]) -> Tuple[AbstractSet[K], bool]:
        """Returns all values that could succeed this prefix and remain in the trie,
        and whether the prefix itself is in the trie.

        For example, if the trie contains "abc", "ade", "bfg",
        a prefix of "a" would give the answer ({"b", "d"}, False).
        """
        node = self._prefix_to_node(prefix)
        if node is None:
            raise KeyError
        return node.children.keys(), node.is_terminal

    def prefix_items(self, prefix: Sequence[K]) -> Iterator[Sequence[K]]:
        """Return all elements in the trie with the given prefix.

        Returned elments are always tuples, even if other kinds of Sequences were added.
        """
        node = self._prefix_to_node(prefix)
        if node is None:
            return
        for item in self._iter_from_node(node):
            yield tuple(prefix) + item

    def prefix_count(self, prefix: Sequence[K]) -> int:
        """Count all elements in the trie that have the given prefix."""
        node = self._prefix_to_node(prefix)
        if node is None:
            return 0
        return self._count_from_node(node)

    def _prefix_to_node(self, prefix: Sequence[K]) -> Optional[TrieSetNode[K]]:
        node = self.root
        for elem in prefix:
            child_node = node.children.get(elem)
            if child_node is None:
                return None
            node = child_node
        return node

    def _iter_from_node(self, node: TrieSetNode[K]) -> Iterator[Tuple[K, ...]]:
        stack = [_SearchState[K](node, prefix=())]
        while stack:
            state = stack.pop()
            if state.node.is_terminal:
                yield state.prefix
            for elem, child_node in state.node.children.items():
                stack.append(_SearchState(child_node, prefix=state.prefix + (elem,)))

    def _count_from_node(self, node: TrieSetNode[K]) -> int:
        result = 0
        stack = [_SearchState(node, prefix=())]
        while stack:
            state = stack.pop()
            if state.node.is_terminal:
                result += 1
            for child_node in state.node.children.values():
                stack.append(_SearchState(child_node, prefix=()))
        return result


class TrieMap(MutableMapping[Sequence[K], V]):
    def __init__(self, iterable: Iterable[Tuple[Sequence[K], V]] = ()):
        # Intended to be a recurisve type, but mypy doesn't support that
        self.root = TrieMapNode[K, V](value=MISSING_SENTINEL, children={})
        for k, v in iterable:
            self[k] = v

    def __getitem__(self, key: Sequence[K]) -> V:
        descendant = self.root.find(key)
        if descendant is None or descendant.value is MISSING_SENTINEL:
            raise KeyError
        assert not isinstance(descendant.value, MissingSentinel)
        return descendant.value

    def __setitem__(self, key: Sequence[K], value: V) -> None:
        descendant = self.root.add(key)
        descendant.value = value

    def __delitem__(self, key: Sequence[K]) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Sequence[K]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@dataclass
class _CompressedNode(Generic[K]):
    __slots__ = ("is_terminal", "children")
    # TODO: In the future, we can turn Trie into a mapping rather than a set
    # by storing arbitrary values here.
    is_terminal: bool
    children: Dict[Tuple[K, ...], "_CompressedNode[K]"]


@dataclass
class _SearchStateForCompression(Generic[K]):
    __slots__ = ("elem", "is_terminal", "remaining_children", "completed_children")

    elem: Optional[K]
    is_terminal: bool
    remaining_children: Dict[K, "TrieSetNode[K]"]
    completed_children: Dict[Tuple[K, ...], "_CompressedNode[K]"]


@dataclass
class _CompressedSearchState(Generic[K]):
    __slots__ = ("node", "prefix")

    node: _CompressedNode
    prefix: Tuple[K, ...]


class CompressedTrie(AbstractSet[Sequence[K]]):
    """Takes a Trie and merges lines of children without splits into one node.

    For example, given [1, 2, 3, 4, 5] and [1, 2, 3, 6] as elements,
    the original Trie would look like this:
      . -1-> . -2-> . -3-> . -4-> . -5-> .
                           . -6-> .
    CompressedTrie represents it this way instead:
      . -1,2,3-> . -4,5-> .
                 . -6-> .
    """

    def __init__(self, trie: Trie[K]):
        stack = [
            _SearchStateForCompression(
                None, trie.root.is_terminal, dict(trie.root.children), {}
            )
        ]
        while stack:
            top = stack[-1]
            if top.remaining_children:
                next_key, next_value = top.remaining_children.popitem()
                stack.append(
                    _SearchStateForCompression(
                        next_key,
                        next_value.is_terminal,
                        remaining_children=dict(next_value.children),
                        completed_children={},
                    )
                )
            else:
                stack.pop()
                if not stack:
                    self.root = _CompressedNode(top.is_terminal, top.completed_children)
                    continue

                if not top.is_terminal and len(top.completed_children) == 1:
                    path, only_child = top.completed_children.popitem()
                    key = (top.elem,) + path
                    new_node = only_child
                else:
                    key = (top.elem,)
                    new_node = _CompressedNode(top.is_terminal, top.completed_children)
                stack[-1].completed_children[key] = new_node

    def __iter__(self) -> Iterator[Sequence[K]]:
        """Returned elments are always tuples, even if other kinds of Sequences were added."""
        return self._iter_from_node(self.root)

    def __len__(self) -> int:
        return self._count_from_node(self.root)

    def __contains__(self, seq: Any) -> bool:
        if not isinstance(  # pylint: disable=isinstance-second-argument-not-valid-type
            seq, Sequence
        ):
            return False
        node = self._prefix_to_node(seq)
        if node is None:
            return False
        return node.is_terminal

    # Below are methods not in the interface for AbstractSet
    def prefix_next(
        self, prefix: Sequence[K]
    ) -> Tuple[AbstractSet[Tuple[K, ...]], bool]:
        """Returns the set of completions for the given prefix,
        such that if the prefix is extended with a completion,
        that extended prefix is a prefix of some element in the trie.
        Also returns whether the provided prefix itself is in the trie.

        `prefix` can be either empty, or formed by appending a valid prefix
        with a result of `prefix_next` on that valid prefix. It cannot be an
        arbitrary sequence, even if that sequence is a prefix of an element
        in the trie.

        For all completions `c` and `prefix + c[:i]` where `i < len(c) - 1`,
        all elements of the trie which have a prefix of `prefix + c[:i]` also
        have `prefix + c` as a prefix.

        The practical use of this function is to expose the internal
        structure of the compressed trie as a tree.
        """
        node = self._prefix_to_node(prefix)
        if node is None:
            raise KeyError
        return node.children.keys(), node.is_terminal

    def _prefix_to_node(self, prefix: Sequence[K]) -> Optional[_CompressedNode[K]]:
        node = self.root
        remaining_prefix = prefix
        while remaining_prefix:
            extension_lengths = set(len(k) for k in node.children)
            for length in extension_lengths:
                child_node = node.children.get(tuple(remaining_prefix[:length]))
                if child_node is not None:
                    node = child_node
                    remaining_prefix = remaining_prefix[length:]
                    break
            else:
                return None
        return node

    def _iter_from_node(self, node: _CompressedNode[K]) -> Iterator[Tuple[K, ...]]:
        stack = [_CompressedSearchState(node, prefix=())]
        while stack:
            state = stack.pop()
            if state.node.is_terminal:
                yield state.prefix
            for elem, child_node in state.node.children.items():
                stack.append(
                    _CompressedSearchState(child_node, prefix=state.prefix + elem)
                )

    def _count_from_node(self, node: _CompressedNode[K]) -> int:
        result = 0
        stack = [_CompressedSearchState(node, prefix=())]
        while stack:
            state = stack.pop()
            if state.node.is_terminal:
                result += 1
            for child_node in state.node.children.values():
                stack.append(_CompressedSearchState(child_node, prefix=()))
        return result
