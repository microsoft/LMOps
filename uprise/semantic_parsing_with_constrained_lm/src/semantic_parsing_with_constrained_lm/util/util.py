# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from more_itertools import chunked

_A = TypeVar("_A")
_N = TypeVar("_N", int, float)

Infinity = float("inf")


def identity(x: _A) -> _A:
    """Returns `x`."""
    return x


# pylint: disable=W0613
def no_op(*args: Any, **kwargs: Any) -> None:
    """Does nothing."""
    return


def batch_by(
    data: List[_A], batch_size: int, key: Callable[[_A], _N]
) -> List[List[_A]]:
    """
    Sort by key and then chunk, so that similarly sized data are grouped
    together, which makes storing and processing padded batches less wasteful.
    """
    return list(chunked(sorted(list(data), key=key), batch_size))


def mean(xs: Iterable[_N]) -> float:
    """Numerically-stable streaming average. O(1) memory."""
    result = 0.0
    for n, x in enumerate(iter(xs)):
        d = 1 / (1 + n)
        result += x * d - result * d
    return result


def flatten(list_in: Iterable[Iterable[_A]]) -> Iterable[_A]:
    return list(itertools.chain.from_iterable(list_in))


def spans(length: int) -> Iterable[Tuple[int, int]]:
    """Returns all spans between 0 and length.

    For example, given length=3, returns:
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)"""

    for start in range(length):
        for end in range(start + 1, length + 1):
            yield (start, end)


def head_or_random(list_in: List[_A], randomize: bool) -> _A:
    if randomize:
        return random.choice(list_in)
    else:
        return list_in[0]


def maybe_randomize(list_in: List[_A], randomize: bool) -> List[_A]:
    """
    If randomize is true, return a shuffled copy of the list.
    Otherwise, just return the list.
    """
    if randomize:
        return random.sample(list_in, len(list_in))
    else:
        return list_in


def extract_tag(tag: str) -> Dict[str, str]:
    """Extract a tag string into the right format for GQL queries."""
    if "=" in tag:
        # if tag contains multiple "=", only split on the first
        (name, content) = tag.split("=", 1)
        return {"name": name, "content": content}
    else:
        return {"name": tag}


def extract_tags(tags: Iterable[str]) -> List[Dict[str, str]]:
    return [extract_tag(tag) for tag in tags]


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_K1 = TypeVar("_K1")
_K2 = TypeVar("_K2")


class IteratorGenerator(Iterable[_T1]):
    """
    A wrapper around a function that returns an iterator. Calling iter() on
    one of these allows you to "reset" the iterator returned by the function.
    """

    def __init__(self, f: Callable[[], Iterator[_T1]]):
        self.f = f

    def __iter__(self) -> Iterator[_T1]:
        yield from self.f()


def cross_product(*iterables: Iterable[_T1]) -> Iterator[Tuple[_T1, ...]]:
    """
    Given a list of callables which produce generators, return a generator representing
    the cross product of those generators.

    Unlike itertools.product, this function is lazy and does not first convert all the Iterables into lists.
    Instead, it calls `iter` on each one at the start of each iteration through it.
    """
    if not iterables:
        # base case
        yield ()
    else:
        heads, *tails = iterables
        for head in heads:
            for tail in cross_product(*tails):
                yield (head,) + tail


def aligned_zip(
    left_iter: Iterable[_T1],
    right_iter: Iterable[_T2],
    left_key_fn: Callable[[_T1], _K1],
    right_key_fn: Callable[[_T2], _K2],
    increment_left: Callable[[_K1, _K2], bool],
) -> Iterator[Tuple[_T1, _T2]]:
    """
    Zip `left_iter` and `right_iter` together as long as
    `left_key_fn(left_item) == right_key_fn(right_item)`.

    If they are not equal, consult `increment_left` to see if we should
    increment `left_iter` or `right_iter` until the two of them match.

    This is useful when `left_iter` and `right_iter` are both sorted in terms
    of their keys, but on occasion, either may be missing some items that the
    other has. Then we can skip items from the other until both are
    synchronized again.

    See tests for some examples for how to use this function.
    """

    left_iter = iter(left_iter)
    right_iter = iter(right_iter)

    left_item: _T1 = None
    left_key: _K1 = None
    right_item: _T2 = None
    right_key: _K2 = None

    def next_left():
        nonlocal left_item, left_key
        left_item = next(left_iter)
        left_key = left_key_fn(left_item)

    def next_right():
        nonlocal right_item, right_key
        right_item = next(right_iter)
        right_key = right_key_fn(right_item)

    try:
        next_left()
        next_right()
        while True:
            if left_key == right_key:
                yield (left_item, right_item)
                next_left()
                next_right()
            elif increment_left(left_key, right_key):
                next_left()
            else:
                next_right()
    except StopIteration:
        pass


def bisect_right(
    lst: Sequence[_A],
    elem: _A,
    key: Callable[[_A], Any] = identity,
    begin: int = 0,
    end: Optional[int] = None,
):
    """Similar to bisect.bisect_right in the standard library, but with a key function.

    Locates the insertion point for `elem` in `lst` to maintain sorted order.
    If `elem` is already present in `lst`, then the insertion point will be after (to the right of) any existing entries.

    The `key` argument has the same meaning as that of `max`, `min`, and `sorted` in the standard library.
    When we need to compare two elements `x` and `y`, we compare `key(x)` and `key(y)`.
    """
    assert begin >= 0
    if end is None:
        end = len(lst)
    else:
        assert end <= len(lst)

    while begin < end:
        mid = (begin + end) // 2
        if key(elem) < key(lst[mid]):
            end = mid
        else:
            begin = mid + 1
    return begin
