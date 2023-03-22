# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections
import inspect
import os
from typing import Dict, List, Tuple

import pytest
import yaml
from lark import Tree

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.generate import (
    generate_from_grammar_and_nonterminal,
    generate_from_parse_tree,
    generate_synchronously,
    parse_and_render,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar, parse_string
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.scfg import (
    SCFG,
    convert_to_lark_rule,
    get_nonterminal_ordering,
)


def expansion_to_values(expansion: Expansion) -> Tuple[str, ...]:
    """Given an expansion, turn it into a tuple of underlying strings."""
    return tuple([token.value for token in expansion])


def expansion_to_string(expansion: Expansion) -> str:
    """Given an expansion, turn it into a string"""
    return "".join(expansion_to_values(expansion))


@pytest.fixture(scope="module", name="scfg")
def example_scfg():
    return SCFG.from_folder(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_grammar")
    )


@pytest.fixture(scope="module", name="scfg2")
def example_scfg2():
    return SCFG.from_folder(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_grammar_2")
    )


@pytest.fixture(scope="module", name="parser")
def create_parser():
    return get_scfg_parser("start_for_test")


def test_get_nonterminal_ordering(parser):
    assert get_nonterminal_ordering(parse_string(parser, '"hi" x "bob" y x')) == {
        ("x", 0): 0,
        ("x", 1): 2,
        ("y", 0): 1,
    }


def test_convert_to_lark_rule(parser):
    assert (
        convert_to_lark_rule(parse_string(parser, '"hi" x "bob" y'))
        == '"hi"i x "bob"i y'
    )


def test_scfg(scfg):
    def _helper1(d: Dict[str, List[Expansion]]) -> Dict[str, List[Tuple[str, ...]]]:
        """
        Given a dictionary where the values are lists of Expansion, return a new
        dictionary where the Expansion are converted into Lists of strings.
        """
        new_d = {}
        for key in d:
            new_d[key] = [expansion_to_values(expansion) for expansion in d[key]]
        return new_d

    def _helper2(d: Dict[str, Expansion]) -> Dict[str, Tuple[str, ...]]:
        """
        Given a dictionary where the values are lists of Expansion, return a new
        dictionary where the Expansion are converted into Lists of strings.
        """
        new_d = {}
        for key in d:
            new_d[key] = expansion_to_values(d[key])
        return new_d

    assert _helper1(scfg.plan_grammar_keyed_by_nonterminal) == {
        "start": [("create",)],
        "create": [
            ('"describe"', '" ask"', '" (create"', '" Event"', '")"', "with_person"),
            (
                '"describe"',
                '" ask"',
                '" (create"',
                '" Event"',
                '")"',
                "with_person",
                "and_person",
                "at_place",
            ),
            (
                '"describe"',
                '" ask"',
                '" (create"',
                '" Event"',
                '")"',
                "with_person",
                "and_person",
            ),
            (
                '"describe"',
                '" ask"',
                '" (create"',
                '" Event"',
                '")"',
                "with_person",
                "at_place",
            ),
        ],
        "with_person": [('" with attendees includes do the Recipient"', "person")],
        "and_person": [('" includes do the Recipient"', "person")],
        "at_place": [('" do the Place "', '"/"', '" at [x]"',)],
        "person": [('" \\"James\\""',), ('" \\"Julie\\""',)],
    }

    assert _helper1(scfg.utterance_grammar_keyed_by_nonterminal) == {
        "start": [("create",)],
        "create": [
            ("set_up", "meeting", "with_person"),
            ('"please"', '" find time"', "with_person"),
            ('" find time"', "with_person"),
            ('"please"', '" find time"', "with_person", "and_person", "at_place"),
            ('" find time"', "with_person", "and_person", "at_place"),
            ('"please"', '" find time"', "with_person", "and_person"),
            ('" find time"', "with_person", "and_person"),
            ('"please"', '" find time"', "with_person", "at_place"),
            ('" find time"', "with_person", "at_place"),
        ],
        "with_person": [('" with"', "person")],
        "and_person": [('" and"', "person")],
        "at_place": [('" there"',)],
        "person": [('" James"',), ('" Julie"',)],
        "meeting": [('" meeting"',), ('" time"',)],
        "set_up": [("",), ('"set up a"',)],
    }

    assert _helper2(scfg.plan_grammar_keyed_by_alias) == {
        "start_start_0": ("create",),
        "create_create_0_create_1_create_2": (
            '"describe"',
            '" ask"',
            '" (create"',
            '" Event"',
            '")"',
            "with_person",
        ),
        "create_create_3_create_4": (
            '"describe"',
            '" ask"',
            '" (create"',
            '" Event"',
            '")"',
            "with_person",
            "and_person",
            "at_place",
        ),
        "create_create_5_create_6": (
            '"describe"',
            '" ask"',
            '" (create"',
            '" Event"',
            '")"',
            "with_person",
            "and_person",
        ),
        "create_create_7_create_8": (
            '"describe"',
            '" ask"',
            '" (create"',
            '" Event"',
            '")"',
            "with_person",
            "at_place",
        ),
        "with_person_with_person_0": (
            '" with attendees includes do the Recipient"',
            "person",
        ),
        "and_person_and_person_0": ('" includes do the Recipient"', "person"),
        "at_place_at_place_0": ('" do the Place "', '"/"', '" at [x]"'),
        "person_person_0": ('" \\"James\\""',),
        "person_person_1": ('" \\"Julie\\""',),
    }

    assert _helper2(scfg.utterance_grammar_keyed_by_alias) == {
        "start_0": ("create",),
        "create_0": ("set_up", "meeting", "with_person"),
        "create_1": ('"please"', '" find time"', "with_person"),
        "create_2": ('" find time"', "with_person"),
        "create_3": (
            '"please"',
            '" find time"',
            "with_person",
            "and_person",
            "at_place",
        ),
        "create_4": ('" find time"', "with_person", "and_person", "at_place"),
        "create_5": ('"please"', '" find time"', "with_person", "and_person"),
        "create_6": ('" find time"', "with_person", "and_person"),
        "create_7": ('"please"', '" find time"', "with_person", "at_place"),
        "create_8": ('" find time"', "with_person", "at_place"),
        "with_person_0": ('" with"', "person"),
        "and_person_0": ('" and"', "person"),
        "at_place_0": ('" there"',),
        "person_0": ('" James"',),
        "person_1": ('" Julie"',),
        "meeting_0": ('" meeting"',),
        "meeting_1": ('" time"',),
        "set_up_0": ("",),
        "set_up_1": ('"set up a"',),
    }

    assert scfg.plan_nonterminal_indices_by_alias == {
        "start_start_0": {("create", 0): 0},
        "create_create_0_create_1_create_2": {("with_person", 0): 0},
        "create_create_3_create_4": {
            ("with_person", 0): 0,
            ("and_person", 0): 1,
            ("at_place", 0): 2,
        },
        "create_create_5_create_6": {("with_person", 0): 0, ("and_person", 0): 1},
        "create_create_7_create_8": {("with_person", 0): 0, ("at_place", 0): 1},
        "with_person_with_person_0": {("person", 0): 0},
        "and_person_and_person_0": {("person", 0): 0},
        "at_place_at_place_0": {},
        "person_person_0": {},
        "person_person_1": {},
    }

    assert scfg.utterance_nonterminal_indices_by_alias == {
        "start_0": {("create", 0): 0},
        "create_0": {("set_up", 0): 0, ("meeting", 0): 1, ("with_person", 0): 2},
        "create_1": {("with_person", 0): 0},
        "create_2": {("with_person", 0): 0},
        "create_3": {("with_person", 0): 0, ("and_person", 0): 1, ("at_place", 0): 2},
        "create_4": {("with_person", 0): 0, ("and_person", 0): 1, ("at_place", 0): 2},
        "create_5": {("with_person", 0): 0, ("and_person", 0): 1},
        "create_6": {("with_person", 0): 0, ("and_person", 0): 1},
        "create_7": {("with_person", 0): 0, ("at_place", 0): 1},
        "create_8": {("with_person", 0): 0, ("at_place", 0): 1},
        "with_person_0": {("person", 0): 0},
        "and_person_0": {("person", 0): 0},
        "at_place_0": {},
        "person_0": {},
        "person_1": {},
        "meeting_0": {},
        "meeting_1": {},
        "set_up_0": {},
        "set_up_1": {},
    }

    assert scfg.utterance_alias_to_plan_alias == {
        "start_0": ["start_start_0"],
        "create_0": ["create_create_0_create_1_create_2"],
        "create_1": ["create_create_0_create_1_create_2"],
        "create_2": ["create_create_0_create_1_create_2"],
        "create_3": ["create_create_3_create_4"],
        "create_4": ["create_create_3_create_4"],
        "create_5": ["create_create_5_create_6"],
        "create_6": ["create_create_5_create_6"],
        "create_7": ["create_create_7_create_8"],
        "create_8": ["create_create_7_create_8"],
        "with_person_0": ["with_person_with_person_0"],
        "and_person_0": ["and_person_and_person_0"],
        "at_place_0": ["at_place_at_place_0"],
        "person_0": ["person_person_0"],
        "person_1": ["person_person_1"],
    }

    assert scfg.plan_alias_to_utterance_alias == {
        "start_start_0": ["start_0"],
        "create_create_0_create_1_create_2": ["create_0", "create_1", "create_2"],
        "create_create_3_create_4": ["create_3", "create_4"],
        "create_create_5_create_6": ["create_5", "create_6"],
        "create_create_7_create_8": ["create_7", "create_8"],
        "with_person_with_person_0": ["with_person_0"],
        "and_person_and_person_0": ["and_person_0"],
        "at_place_at_place_0": ["at_place_0"],
        "person_person_0": ["person_0"],
        "person_person_1": ["person_1"],
    }

    assert scfg.utterance_lark.parser.parse(
        "please find time with julie there"
    ) == Tree(
        "start_0",
        [
            Tree(
                "create_7",
                [Tree("with_person_0", [Tree("person_1", [])]), Tree("at_place_0", [])],
            )
        ],
    )

    assert scfg.plan_lark.parser.parse(
        'describe ask (create Event) with attendees includes do the Recipient "James" do the Place / at [x]'
    ) == Tree(
        "start_start_0",
        [
            Tree(
                "create_create_7_create_8",
                [
                    Tree("with_person_with_person_0", [Tree("person_person_0", [])]),
                    Tree("at_place_at_place_0", []),
                ],
            )
        ],
    )

    assert (
        scfg.utterance_lark.grammar
        == inspect.cleandoc(
            """and_person: " and"i person -> and_person_0
        at_place: " there"i -> at_place_0
        create: set_up meeting with_person -> create_0
        |"please"i " find time"i with_person -> create_1
        |" find time"i with_person -> create_2
        |"please"i " find time"i with_person and_person at_place -> create_3
        |" find time"i with_person and_person at_place -> create_4
        |"please"i " find time"i with_person and_person -> create_5
        |" find time"i with_person and_person -> create_6
        |"please"i " find time"i with_person at_place -> create_7
        |" find time"i with_person at_place -> create_8
        meeting: " meeting"i -> meeting_0
        |" time"i -> meeting_1
        person: " James"i -> person_0
        |" Julie"i -> person_1
        set_up:  -> set_up_0
        |"set up a"i -> set_up_1
        start: create -> start_0
        with_person: " with"i person -> with_person_0"""
        )
        + "\n"
    )

    assert (
        scfg.plan_lark.grammar
        == inspect.cleandoc(
            """and_person: " includes do the Recipient" person -> and_person_and_person_0
        at_place: " do the Place " "/" " at [x]" -> at_place_at_place_0
        create: "describe" " ask" " (create" " Event" ")" with_person -> create_create_0_create_1_create_2
        |"describe" " ask" " (create" " Event" ")" with_person and_person at_place -> create_create_3_create_4
        |"describe" " ask" " (create" " Event" ")" with_person and_person -> create_create_5_create_6
        |"describe" " ask" " (create" " Event" ")" with_person at_place -> create_create_7_create_8
        person: " \\"James\\"" -> person_person_0
        |" \\"Julie\\"" -> person_person_1
        start: create -> start_start_0
        with_person: " with attendees includes do the Recipient" person -> with_person_with_person_0"""
        )
        + "\n"
    )


def load_plan_utterance_pairs(test_filename: str):
    return yaml.load(
        open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), test_filename,),
            "r",
        ),
        Loader=yaml.BaseLoader,
    )


def test_round_trip(scfg: SCFG):
    def assert_round_trips(grammar: SCFG, test_filename: str, topological_render: bool):
        plan_utterance_pairs = load_plan_utterance_pairs(test_filename)
        for plan in plan_utterance_pairs:
            for utterance in plan_utterance_pairs[plan]:
                if not topological_render:
                    plan_tree = grammar.plan_lark.parser.parse(plan)
                    generated_utterances = generate_from_parse_tree(
                        plan_tree, grammar, source_is_plan=True, randomize=False
                    )

                    assert utterance in [u.render() for u in generated_utterances]

                utterance_tree = grammar.utterance_lark.parser.parse(utterance)
                generated_plans = list(
                    generate_from_parse_tree(
                        utterance_tree, grammar, source_is_plan=False, randomize=False
                    )
                )

                assert len(generated_plans) == 1

                if topological_render:
                    assert generated_plans[0].render_topological() == plan
                else:
                    assert generated_plans[0].render() == plan

    assert_round_trips(scfg, "test_utterances_and_plans.yml", topological_render=False)
    assert_round_trips(
        scfg, "test_utterances_and_plans_topological.yml", topological_render=True
    )

    regex_grammar = SCFG.from_folder(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_grammar_parse")
    )
    assert_round_trips(
        regex_grammar, "test_utterances_and_plans.yml", topological_render=False
    )
    assert_round_trips(
        regex_grammar,
        "test_utterances_and_plans_topological.yml",
        topological_render=True,
    )


def generate_synchronously_helper(scfg, randomize):
    return [
        (utterance_tree.render(), plan_tree.render())
        for utterance_tree, plan_tree in generate_synchronously(
            scfg, randomize=randomize
        )
    ]


def test_synchronous_generation(scfg: SCFG):
    plan_utterance_pairs = set(
        (utterance, plan)
        for plan, utterances in load_plan_utterance_pairs(
            "test_utterances_and_plans.yml"
        ).items()
        for utterance in utterances
    )
    generated_plan_utterance_pairs = set(
        generate_synchronously_helper(scfg, randomize=False)
    )

    assert plan_utterance_pairs <= generated_plan_utterance_pairs

    def generate_one_side(name):
        return set(
            tree.render()
            for tree in generate_from_grammar_and_nonterminal(
                getattr(scfg, f"{name}_grammar_keyed_by_alias"),
                getattr(scfg, f"{name}_nonterminal_to_aliases"),
                randomize=False,
            )
        )

    generated_plans = generate_one_side("plan")
    synchronous_generated_plans = set(
        plan for _, plan in generated_plan_utterance_pairs
    )
    assert generated_plans == synchronous_generated_plans

    generated_utterances = generate_one_side("utterance")
    synchronous_generated_utterances = set(
        utterance for utterance, _ in generated_plan_utterance_pairs
    )
    assert generated_utterances == synchronous_generated_utterances


def test_synchronous_generation_sample(scfg: SCFG):
    all_plan_utterance_pairs = generate_synchronously_helper(scfg, randomize=False)

    randomized = False
    for _ in range(100):
        randomized_plan_utterance_pairs = generate_synchronously_helper(
            scfg, randomize=True
        )
        if (randomized_plan_utterance_pairs != all_plan_utterance_pairs) and (
            collections.Counter(randomized_plan_utterance_pairs)
            == collections.Counter(all_plan_utterance_pairs)
        ):
            randomized = True
            break
    assert randomized


def test_plan_to_utterance_with_a_regex_in_an_utterance_only_rule():
    grammar_str = """

    # uses two utterance-only rules and one sync rule
    start -> createEventPrefix cmspace createEventSuffix eventPredicate , "describe create event" eventPredicate
    eventPredicate -> !" today" , " today"

    # utterance-only rules
    createEventPrefix 1> !"create"
    createEventSuffix 1> !"an event"
    cmspace 1> !/ /
    """

    grammar = SCFG(PreprocessedGrammar.from_line_iter(grammar_str.splitlines()))
    utterance = "create an event today"
    (plan,) = list(parse_and_render(grammar, utterance, source_is_plan=False))
    assert utterance in parse_and_render(grammar, plan, source_is_plan=True)
