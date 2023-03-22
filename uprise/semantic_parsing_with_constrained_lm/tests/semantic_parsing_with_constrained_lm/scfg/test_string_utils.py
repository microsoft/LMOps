# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.utils import is_skippable
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.string_utils import detokenize


def test_is_comment():
    assert is_skippable("#hi")
    assert is_skippable("")
    assert not is_skippable("hi")


def test_detokenize():
    assert (
        detokenize(["find", "Event", "time", ".", "results", "chris", "'s", "car"])
        == "find Event time . results chris 's car"
    )

    assert detokenize(["f", "(", "x", ",", "y", ")"], with_treebank=False) == "f(x,y)"
