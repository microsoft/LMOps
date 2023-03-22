# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import textwrap
from dataclasses import dataclass

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm import fewshot


@dataclass
class Datum:
    a: str
    b: str
    c: str


def test_prompt_builder():
    prompt_builder = fewshot.PromptBuilder(
        problem_spec=fewshot.ProblemSpec(
            input_fields=frozenset(["a", "b"]), output_field="c"
        ),
        preamble="Hello!\n",
        input_field_order=["a", "b"],
        field_to_adornment={
            "a": fewshot.Adornment("a = ", "\n"),
            "b": fewshot.Adornment("b is ", "!\n"),
            "c": fewshot.Adornment("c: ", "\n"),
        },
        datum_adornment=fewshot.Adornment("Here is one datum.\n", ""),
        separator="--\n",
    )

    assert prompt_builder.assemble(
        train_data=[
            Datum(a="foo", b="bar", c="c1"),
            Datum(a="alice", b="bob", c="carol"),
        ],
        test_datum=None,
    ) == textwrap.dedent(
        """\
            Hello!
            --
            Here is one datum.
            a = foo
            b is bar!
            c: c1
            --
            Here is one datum.
            a = alice
            b is bob!
            c: carol
            """
    )

    assert prompt_builder.assemble(
        train_data=[
            Datum(a="foo", b="bar", c="c1"),
            Datum(a="alice", b=None, c="carol"),
        ],
        test_datum=Datum(a="a", b="b", c="c3"),
    ) == textwrap.dedent(
        """\
            Hello!
            --
            Here is one datum.
            a = foo
            b is bar!
            c: c1
            --
            Here is one datum.
            a = alice
            c: carol
            --
            Here is one datum.
            a = a
            b is b!
            c: """
    )
