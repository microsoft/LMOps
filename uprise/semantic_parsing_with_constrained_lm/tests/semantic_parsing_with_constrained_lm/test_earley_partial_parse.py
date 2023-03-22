# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=redefined-outer-name
from typing import Callable, List, TypeVar

import pytest
import torch

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley_partial_parse import (
    EarleyPartialParse,
    GrammarTokenizerInfo,
    UTF8EarleyPartialParse,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.search import PartialParse

# This grammar can generate:
#   abcA
#   abcB
#   abcC
#   abcDE
SIMPLE_GRAMMAR = """
start -> a, a
start -> b, b
start -> c, c
a -> !"a" !"bcA", "abcA"
a -> !"ab" !"cB", "abcB"
b -> !"abc" !"C", "abcC"
b -> c !"DE", "abcDE"
c -> !"a" !"bc", "abc"
"""


@pytest.fixture(scope="module")
def simple_grammar() -> PreprocessedGrammar:
    return PreprocessedGrammar.from_line_iter(SIMPLE_GRAMMAR.splitlines())


S = TypeVar("S", str, bytes)


def make_helpers(vocab: List[S]):
    token_to_id = {token: i for i, token in enumerate(vocab)}

    def compare(
        partial_parse: PartialParse, expected_tokens: List[S], expected_can_end: bool,
    ) -> None:
        tokens, can_end = partial_parse.allowed_next(
            torch.arange(len(vocab), dtype=torch.long)
        )
        assert set(tokens.tolist()) == set(token_to_id[t] for t in expected_tokens)
        assert can_end == expected_can_end

    def append(partial_parse: PartialParse, token: S) -> PartialParse:
        return partial_parse.append(token_to_id[token])

    return compare, append


parameterize_partial_parse_implementations = pytest.mark.parametrize(
    argnames=["initial_factory"],
    argvalues=[
        (
            lambda vocab, preproc_grammar, input_utterance: EarleyPartialParse.initial(
                GrammarTokenizerInfo.from_tokens_list(
                    vocab, preproc_grammar, for_plans=False
                ),
                input_utterance,
            ),
        ),
        (
            lambda vocab, preproc_grammar, input_utterance: UTF8EarleyPartialParse.initial(
                GrammarTokenizerInfo.from_tokens_list(
                    vocab, preproc_grammar, for_plans=False
                ),
                input_utterance,
            ),
        ),
    ],
    ids=["EarleyPartialParse", "UTF8EarleyPartialParse",],
)


def utf8_initial(
    vocab: List[bytes], preproc_grammar: PreprocessedGrammar, input_utterance: str
) -> UTF8EarleyPartialParse:
    return UTF8EarleyPartialParse.initial(
        GrammarTokenizerInfo.from_utf8_tokens_list(
            vocab, preproc_grammar, for_plans=False
        ),
        input_utterance,
    )


@parameterize_partial_parse_implementations
def test_simple_grammar_single_char_vocab(
    simple_grammar: PreprocessedGrammar,
    initial_factory: Callable[[List[str], PreprocessedGrammar, str], PartialParse],
) -> None:
    single_char_vocab = ["a", "b", "c", "A", "B", "C", "D", "E"]
    initial = initial_factory(single_char_vocab, simple_grammar, "")
    compare, append = make_helpers(single_char_vocab)

    compare(initial, ["a"], False)
    pp_a = append(initial, "a")
    compare(pp_a, ["b"], False)
    pp_ab = append(pp_a, "b")
    compare(pp_ab, ["c"], False)
    pp_abc = append(pp_ab, "c")
    compare(pp_abc, ["A", "B", "C", "D"], True)
    pp_abcA = append(pp_abc, "A")
    compare(pp_abcA, [], True)
    pp_abcB = append(pp_abc, "B")
    compare(pp_abcB, [], True)
    pp_abcC = append(pp_abc, "C")
    compare(pp_abcC, [], True)
    pp_abcD = append(pp_abc, "D")
    compare(pp_abcD, ["E"], False)
    pp_abcDE = append(pp_abcD, "E")
    compare(pp_abcDE, [], True)


@parameterize_partial_parse_implementations
def test_simple_grammar_multi_char_vocab(
    simple_grammar: PreprocessedGrammar,
    initial_factory: Callable[[List[str], PreprocessedGrammar, str], PartialParse],
) -> None:
    multi_char_vocab = [
        "a",
        "b",
        "c",
        "ab",
        "bc",
        "cD",
        "A",
        "B",
        "C",
        "D",
        "E",
        "DE",
        "abcD",
        "abcdef",
    ]
    initial = initial_factory(multi_char_vocab, simple_grammar, "")
    compare, append = make_helpers(multi_char_vocab)

    compare(initial, ["a", "ab", "abcD"], False)
    pp_a = append(initial, "a")
    compare(pp_a, ["b", "bc"], False)
    pp_ab_1 = append(initial, "ab")
    compare(pp_ab_1, ["c", "cD"], False)
    pp_ab_2 = append(pp_a, "b")
    compare(pp_ab_2, ["c", "cD"], False)
    pp_abc_1 = append(pp_ab_1, "c")
    compare(pp_abc_1, ["A", "B", "C", "D", "DE"], True)
    pp_abc_2 = append(pp_a, "bc")
    compare(pp_abc_2, ["A", "B", "C", "D", "DE"], True)
    pp_abcA_1 = append(pp_abc_1, "A")
    compare(pp_abcA_1, [], True)
    pp_abcA_2 = append(pp_abc_2, "A")
    compare(pp_abcA_2, [], True)
    pp_abcD_1 = append(initial, "abcD")
    compare(pp_abcD_1, ["E"], False)
    pp_abcD_2 = append(pp_abc_1, "D")
    compare(pp_abcD_2, ["E"], False)
    pp_abcDE_1 = append(pp_abcD_1, "E")
    compare(pp_abcDE_1, [], True)
    pp_abcDE_2 = append(pp_abcD_2, "E")
    compare(pp_abcDE_2, [], True)

    # Try again but only with one path
    initial = initial_factory(multi_char_vocab, simple_grammar, "")
    compare(initial, ["a", "ab", "abcD"], False)
    pp_ab = append(initial, "ab")
    compare(pp_ab, ["c", "cD"], False)
    pp_abcD = append(pp_ab, "cD")
    compare(pp_abcD, ["E"], False)
    pp_abcDE = append(pp_abcD, "E")
    compare(pp_abcDE, [], True)


EMOJI_GRAMMAR = """
start -> cold monster, cold " " monster
cold ->    !"\N{snowman}", "snowman"
cold ->    !"\N{snowflake}", "snowflake"
cold ->    !"\N{freezing face}", "freezing face"
monster -> !"\N{ghost}", "ghost"
monster -> !"\N{alien monster}", "alien monster"
monster -> !"\N{biohazard sign}", "biohazard sign"
"""
# UTF-8 encoded:
# snowman: b'\xe2\x98\x83'
# snowflake: b'\xe2\x9d\x84'
# freezing face: b'\xf0\x9f\xa5\xb6'
# ghost: b'\xf0\x9f\x91\xbb'
# alien monster: b'\xf0\x9f\x91\xbe'
# biohazard sign: b'\xe2\x98\xa3'


@pytest.fixture(scope="module")
def emoji_grammar() -> PreprocessedGrammar:
    return PreprocessedGrammar.from_line_iter(EMOJI_GRAMMAR.splitlines())


def test_emoji_grammar_utf8(emoji_grammar: PreprocessedGrammar,) -> None:

    vocab = [
        "\N{snowman}".encode("utf-8"),
        "\N{snowflake}".encode("utf-8"),
        "\N{freezing face}".encode("utf-8"),
        "\N{ghost}".encode("utf-8"),
        "\N{alien monster}".encode("utf-8"),
        "\N{biohazard sign}".encode("utf-8"),
        b"\x98",
        b"\x9d",
        b"\xe2",
        b"\xf0",
        b"\x83\xf0",
        b"\x98\x83",
        b"\x9f\x91",
        b"\xe2\x98",
        b"\x9f\xa5\xb6",
        b"\xf0\x9f\x91",
    ]
    initial = utf8_initial(vocab, emoji_grammar, "")
    compare, append = make_helpers(vocab)

    compare(
        initial,
        [
            "\N{snowman}".encode("utf-8"),
            "\N{snowflake}".encode("utf-8"),
            "\N{freezing face}".encode("utf-8"),
            b"\xe2",
            b"\xe2\x98",
            b"\xf0",
        ],
        False,
    )
    pp_snowman = append(initial, "\N{snowman}".encode("utf-8"))
    after_snowman = [
        "\N{ghost}".encode("utf-8"),
        "\N{alien monster}".encode("utf-8"),
        "\N{biohazard sign}".encode("utf-8"),
        b"\xe2",
        b"\xe2\x98",
        b"\xf0",
        b"\xf0\x9f\x91",
    ]
    compare(pp_snowman, after_snowman, False)
    pp_xf0 = append(initial, b"\xf0")
    compare(pp_xf0, [b"\x9f\xa5\xb6"], False)
    pp_snowman_ghost = append(pp_snowman, "\N{ghost}".encode("utf-8"))
    compare(pp_snowman_ghost, [], True)
    pp_xe2 = append(initial, b"\xe2")
    compare(pp_xe2, [b"\x98", b"\x9d", b"\x98\x83"], False)
    pp_xe2_x98 = append(pp_xe2, b"\x98")
    compare(pp_xe2_x98, [b"\x83\xf0"], False)
    pp_xe2_x98_x83 = append(pp_xe2, b"\x98\x83")
    compare(pp_xe2_x98_x83, after_snowman, False)
    pp_xe2_x98_x83_xf0 = append(pp_xe2_x98, b"\x83\xf0")
    compare(pp_xe2_x98_x83_xf0, [b"\x9f\x91"], False)


REGEX_GRAMMAR = r"""
start -> quoted, quoted
start -> !"'''", "'''"
start -> !"'aA'", "'aA'"
start -> !"'aB'", "'aB'"
start -> !"'aCD'", "'aCD'"
quoted -> !"'" nonquoteplus !"'", "'" nonquoteplus "'"
nonquoteplus -> !/[a-z]/ nonquotestar, /[^"]/ nonquotestar
nonquotestar -> !/[a-z]/ nonquotestar, /[^"]/ nonquotestar
nonquotestar -> #e, #e
"""


@pytest.fixture(scope="module")
def regex_grammar() -> PreprocessedGrammar:
    return PreprocessedGrammar.from_line_iter(REGEX_GRAMMAR.splitlines())


@parameterize_partial_parse_implementations
def test_regex_grammar_single_char_vocab(
    regex_grammar: PreprocessedGrammar,
    initial_factory: Callable[[List[str], PreprocessedGrammar, str], PartialParse],
) -> None:
    single_char_vocab = ["'", "a", "b", "c", "d", "A", "B", "C", "D"]
    initial = initial_factory(single_char_vocab, regex_grammar, "babcc")
    compare, append = make_helpers(single_char_vocab)

    compare(initial, ["'"], False)
    pp_Q = append(initial, "'")
    compare(pp_Q, ["'", "a", "b", "c"], False)
    pp_Qa = append(pp_Q, "a")
    compare(pp_Qa, ["'", "b", "A", "B", "C"], False)
    pp_QaQ = append(pp_Qa, "'")
    compare(pp_QaQ, [], True)
    pp_QQ = append(pp_Q, "'")
    compare(pp_QQ, ["'"], False)
    pp_QaA = append(pp_Qa, "A")
    compare(pp_QaA, ["'"], False)
    pp_Qab = append(pp_Qa, "b")
    compare(pp_Qab, ["'", "c"], False)
    pp_QaC = append(pp_Qa, "C")
    compare(pp_QaC, ["D"], False)
    pp_QabQ = append(pp_Qab, "'")
    compare(pp_QabQ, [], True)


@parameterize_partial_parse_implementations
def test_regex_grammar_multi_char_vocab(
    regex_grammar: PreprocessedGrammar,
    initial_factory: Callable[[List[str], PreprocessedGrammar, str], PartialParse],
) -> None:
    multi_char_vocab = ["'", "''", "a", "aC", "ab", "A", "CD", "d", "de", "def", "'e"]
    initial = initial_factory(multi_char_vocab, regex_grammar, "babcce")
    compare, append = make_helpers(multi_char_vocab)

    compare(initial, ["'", "''", "'e"], False)
    pp_Q = append(initial, "'")
    compare(pp_Q, ["'", "''", "a", "aC", "ab"], False)
    pp_QQ_1 = append(initial, "''")
    compare(pp_QQ_1, ["'"], False)
    pp_QQ_2 = append(initial, "''")
    compare(pp_QQ_2, ["'"], False)
    pp_Qab = append(pp_Q, "ab")
    compare(pp_Qab, ["'"], False)

    # Try with trivial input utterance
    initial = initial_factory(multi_char_vocab, regex_grammar, "d")
    compare, append = make_helpers(multi_char_vocab)
    compare(initial, ["'", "''"], False)
    pp_Q = append(initial, "'")
    compare(pp_Q, ["'", "''", "a", "aC", "d"], False)
    pp_Qd = append(pp_Q, "d")
    compare(pp_Qd, ["'"], False)


EMOJI_REGEX_GRAMMAR = """
start -> quoted, quoted
quoted -> !"'" nonquoteplus !"'", "'" nonquoteplus "'"
nonquoteplus -> !/[^'\N{snowman}]/ nonquotestar, /[^']/ nonquotestar
nonquotestar -> !/[^'\N{snowman}]/ nonquotestar, /[^']/ nonquotestar
nonquotestar -> #e, #e
"""


@pytest.fixture(scope="module")
def emoji_regex_grammar() -> PreprocessedGrammar:
    return PreprocessedGrammar.from_line_iter(EMOJI_REGEX_GRAMMAR.splitlines())


def test_emoji_regex_grammar(emoji_regex_grammar: PreprocessedGrammar,):
    vocab = [
        "\N{snowman}".encode("utf-8"),
        "\N{snowflake}".encode("utf-8"),
        "\N{freezing face}".encode("utf-8"),
        "\N{ghost}".encode("utf-8"),
        "\N{alien monster}".encode("utf-8"),
        "\N{biohazard sign}".encode("utf-8"),
        "'\N{snowman}".encode("utf-8"),
        "'\N{ghost}".encode("utf-8"),
        b"'",
        b"\x98",
        b"\x83",
        b"\x9d",
        b"\xe2",
        b"\xf0",
        b"'\xe2",
        b"\x9f\xa5",
        b"\x9f\xa5\xb6",
        b"\xf0\x9f\x91",
    ]
    initial = utf8_initial(
        vocab, emoji_regex_grammar, "\N{snowman}\N{ghost}\N{alien monster}",
    )
    compare, append = make_helpers(vocab)
    # "'\N{snowman}" is missing because we banned \N{snowman} in the grammar
    compare(initial, ["'\N{ghost}".encode("utf-8"), b"'", b"'\xe2"], False)
    pp_q = append(initial, b"'")
    compare(
        pp_q,
        [
            "\N{ghost}".encode("utf-8"),
            "\N{alien monster}".encode("utf-8"),
            b"\xe2",
            b"\xf0",
            b"\xf0\x9f\x91",
        ],
        False,
    )
