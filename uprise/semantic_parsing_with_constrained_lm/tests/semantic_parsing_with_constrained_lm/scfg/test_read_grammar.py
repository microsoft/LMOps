# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar


def test_from_line_iter():
    with pytest.raises(AssertionError) as excinfo:
        PreprocessedGrammar.from_line_iter(
            ['describe 2> "describe"', 'describe 2> "describe(" ")"']
        )
    assert "Macro describe cannot be defined more than once" in str(excinfo)
    # Doesn't throw.
    PreprocessedGrammar.from_line_iter(['describe 2> "describe"'])
