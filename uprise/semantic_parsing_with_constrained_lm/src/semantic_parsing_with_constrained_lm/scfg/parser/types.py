# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import SCFGToken

Nonterminal = str
# An Alias is just another name for a nonterminal.
Alias = str


Expansion = Tuple[SCFGToken, ...]
