# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.input import SigmaStarTriePosition


def test_sigmastar():
    p = SigmaStarTriePosition[str]()
    (a_1,) = p.scan("a")
    (a_2,) = p.scan("a")
    assert id(a_1) == id(a_2), "scans should be cached and reused"

    (as_1,) = a_1.scan("s")
    (asd,) = as_1.scan("d")
    (asdf,) = asd.scan("f")
    assert asdf.last() == "f"
    assert asdf.prefix() == ["a", "s", "d", "f"]

    (asde,) = asd.scan("e")
    assert asde.prefix() == ["a", "s", "d", "e"]
