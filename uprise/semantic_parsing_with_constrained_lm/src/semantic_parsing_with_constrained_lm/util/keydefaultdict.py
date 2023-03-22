# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Adapted from https://stackoverflow.com/posts/2912455/revisions,
# but with type annotations added.
# pylint: disable=no-member,useless-super-delegation
from typing import Any, Callable, DefaultDict, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class KeyDefaultDict(DefaultDict[K, V]):
    """
    A version of defaultdict whose factory function that constructs a
    default value for a missing key takes that key as an argument.

    >>> d: KeyDefaultDict[int, str] = KeyDefaultDict(lambda k: "/" + str(k) + "/", {0: "zero"})
    >>> d[3] = 'three'
    >>> d[0]
    'zero'
    >>> d[3]
    'three'
    >>> d[4]
    '/4/'
    >>> dict(d)
    {0: 'zero', 3: 'three', 4: '/4/'}
    """

    def __init__(self, default_factory: Callable[[K], V], *args: Any, **kwargs: Any):
        super().__init__(None, *args, **kwargs)
        # store the default_factory in an attribute of a different name, to avoid an inheritance type error
        self.default_key_factory = default_factory

    def __missing__(self, key: K) -> V:
        """
        Overrides the central method of `defaultdict` with one that calls
        `default_key_factory` on `key` instead of calling `default_factory`
        on 0 args.
        """
        if self.default_key_factory is None:
            raise KeyError(key)
        ret = self[key] = self.default_key_factory(key)
        return ret

    def __repr__(self) -> str:
        """Prints `default_key_factory` instead of `default_factory`."""
        return f"{self.__class__.__name__}({self.default_key_factory}, {dict.__repr__(self)})"

    # To avoid E1136 (unsubscriptable-object) pylint errors at call sites
    def __getitem__(self, item: K) -> V:
        return super().__getitem__(item)

    # To avoid E1136 (unsubscriptable-object) pylint errors at call sites
    def __setitem__(self, key: K, value: V) -> None:
        return super().__setitem__(key, value)
