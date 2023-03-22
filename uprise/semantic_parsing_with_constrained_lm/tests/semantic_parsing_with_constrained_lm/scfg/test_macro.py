# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import pytest

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.macro import Macro, eval_expression, expand_macros
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import parse_string
from test_semantic_parsing_with_constrained_lm.scfg.test_scfg import expansion_to_string


@pytest.fixture(scope="module", name="parser")
def create_parser():
    return get_scfg_parser("start_for_test")


@pytest.fixture(scope="module", name="macro_rules")
def def_macro_rules(parser) -> Dict[str, Macro]:
    return {
        "f": parse_string(parser, 'f(x) 2> "(f" x z ")"'),
        "g": parse_string(parser, 'g(y1, y2) 2> "(g" y1 y2 ")"'),
        "h": parse_string(parser, "h(x) 2> f(g(x, x))"),
        "i": parse_string(parser, "i() 2> i1 i2"),
        "j": parse_string(parser, "j 2> j1 j2"),
    }


def test_eval_macro(parser, macro_rules):
    def test_eval_macro_helper(token_to_eval, resulting_rhs):
        token = parse_string(parser, token_to_eval)[0]
        assert expansion_to_string(eval_expression(macro_rules, token)) == resulting_rhs

    test_eval_macro_helper("f(zed)", '"(f"zedz")"')
    test_eval_macro_helper('g(zed, "zedder")', '"(g"zed"zedder"")"')
    test_eval_macro_helper('f(g(zed, "zedder"))', '"(f""(g"zed"zedder"")"z")"')
    test_eval_macro_helper("h(zed)", '"(f""(g"zedzed")"z")"')
    test_eval_macro_helper("i()", "i1i2")
    test_eval_macro_helper("j()", "j1j2")
    test_eval_macro_helper("j", "j")
    test_eval_macro_helper("i", "i")


def test_replace_macros(parser, macro_rules):
    def test_replace_helper(input_rhs, output_rhs):
        rhs = parse_string(parser, input_rhs)
        assert expansion_to_string(expand_macros(macro_rules, rhs)) == output_rhs

    test_replace_helper('"(create Event" f(z) ")"', '"(create Event""(f"zz")"")"')
    test_replace_helper(
        '"(create Event" g(z1, z2) ")"', '"(create Event""(g"z1z2")"")"'
    )
    test_replace_helper(
        '"(create Event" f(g(z1, z2)) ")"', '"(create Event""(f""(g"z1z2")"z")"")"',
    )
    test_replace_helper(
        '"(create Event" h(z) j() i ")"', '"(create Event""(f""(g"zz")"z")"j1j2i")"',
    )
