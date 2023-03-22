# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.trie import Trie
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.search import PartialParse


@dataclass
class TriePartialParse(PartialParse):
    trie: Trie[int]
    tokens: Tuple[int, ...] = ()

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, bool]:
        allowed, is_complete = self.trie.prefix_next(self.tokens)
        return torch.tensor(sorted(allowed), dtype=torch.long), is_complete

    def append(self, token: int) -> "PartialParse":
        """Return a new PartialParse creatoted by appending this token."""
        return dataclasses.replace(self, tokens=self.tokens + (token,))
