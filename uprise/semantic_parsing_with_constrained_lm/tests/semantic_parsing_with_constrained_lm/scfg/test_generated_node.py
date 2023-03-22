# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.generated_node import (
    GeneratedNonterminalNode,
    GeneratedTerminalNode,
)


def test_render():

    #
    # The below graph represents the following rules:
    # e3 -> e1 "/" "(f" "x" e2 ")"
    # e2 -> "x[0] = foo" "/" "[x0].baz"
    # e1 -> "foo" e0 "/" "bar"
    # e0 -> "[x1] = do the action" "/" "do [x1]"
    #

    e0 = GeneratedNonterminalNode(
        (
            GeneratedTerminalNode("[x1] = do the action"),
            GeneratedTerminalNode("/"),
            GeneratedTerminalNode(" do [x1]"),
        )
    )

    e1 = GeneratedNonterminalNode(
        (
            GeneratedTerminalNode("foo"),
            e0,
            GeneratedTerminalNode("/"),
            GeneratedTerminalNode("bar"),
        )
    )

    e2 = GeneratedNonterminalNode(
        (
            GeneratedTerminalNode("[x0] = foo"),
            GeneratedTerminalNode("/"),
            GeneratedTerminalNode(" [x0].baz"),
        )
    )
    e3 = GeneratedNonterminalNode(
        (
            e1,
            GeneratedTerminalNode("/"),
            GeneratedTerminalNode("(f"),
            GeneratedTerminalNode(" x"),
            e2,
            GeneratedTerminalNode(")"),
        )
    )

    assert e3.render_topological_list() == [
        "[x1] = do the action",
        "foo do [x1]",
        "bar",
        "[x0] = foo",
        "(f x [x0].baz)",
    ]
    assert (
        e3.render_topological()
        == "[x1] = do the action / foo do [x1] / bar / [x0] = foo / (f x [x0].baz)"
    )
    assert (
        e3.render() == "foo[x1] = do the action/ do [x1]/bar/(f x[x0] = foo/ [x0].baz)"
    )
