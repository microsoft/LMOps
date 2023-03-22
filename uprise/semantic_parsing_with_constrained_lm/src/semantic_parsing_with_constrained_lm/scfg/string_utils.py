# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable


def detokenize(tokens: Iterable[str], with_treebank: bool = True) -> str:
    """
    Given a list of tokens, join them together into a string.
    with_treebank = True is typically used when rendering utterances, so we don't need to deal with things like
    "andrew's"
    with_treebank = False is typically for rendering express.
    """
    if with_treebank:
        return " ".join(tokens).replace("  ", " ")

    return "".join(tokens)
