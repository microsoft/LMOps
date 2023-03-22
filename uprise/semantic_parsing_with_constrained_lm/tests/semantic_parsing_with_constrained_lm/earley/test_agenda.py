# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.agenda import Agenda, Meta


def test_push_pop():
    a = Agenda()
    z = Meta.zero()
    assert a.push(3, z)
    assert a.push(5, z)
    # duplicate should be ignored
    assert not a.push(3, z)
    assert a.popped == []
    assert a.remaining == [3, 5]
    assert a.pop() == 3
    assert a.popped == [3]
    assert a.remaining == [5]
    # duplicate should be ignored
    assert not a.push(3, z)
    assert a.push(7, z)

    assert a.popped == [3]
    assert a.remaining == [5, 7]

    def it():
        while a:
            yield a.pop()

    assert list(it()) == [5, 7]
