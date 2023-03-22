# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import glob
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Tuple, Union

from cached_property import cached_property

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.macro import Macro, expand_macros
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser, parse_string
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.rule import (
    Rule,
    SyncRule,
    UtteranceRule,
    all_possible_options,
    modify_expansion,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import EntitySchema
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.utils import is_skippable

GrammarRules = Dict[Nonterminal, List[Expansion]]


def find_all_scfg_paths(folder_path: str) -> List[str]:
    """Finds all file paths under the given path with an .scfg extension."""
    return glob.glob(os.path.join(folder_path, "**", "*.scfg"), recursive=True)


@dataclass
class PreprocessedGrammar:
    sync_rules: Dict[Tuple[Nonterminal, Expansion], List[Expansion]]
    utterance_rules: GrammarRules
    schema_annotations: Dict[Nonterminal, EntitySchema]

    @cached_property
    def all_utterance_rules(self) -> GrammarRules:
        """Merges `utterance_rules` with the utterance part of `sync_rules`."""
        result: DefaultDict[Nonterminal, List[Expansion]] = defaultdict(list)
        for (lhs, _plan_rhs), rhs in self.sync_rules.items():
            result[lhs].extend(rhs)
        for lhs, rhs in self.utterance_rules.items():
            result[lhs].extend(rhs)
        return dict(result)

    @cached_property
    def all_plan_rules(self) -> GrammarRules:
        """
        Extracts the plan part of `sync_rules`
        (there is no stand-alone `plan_rules`).
        """
        result: DefaultDict[Nonterminal, List[Expansion]] = defaultdict(list)
        for (lhs, plan_rhs), _ in self.sync_rules.items():
            result[lhs].append(plan_rhs)
        return dict(result)

    @classmethod
    def from_line_iter(cls, grammar_input: Iterable[str]) -> "PreprocessedGrammar":
        """
        Reads an .scfg stream with lines that define:

        Sync rules: nonterminal -> utterance1 | ... | utteranceN , plan
        OR
        Utterance rules: nonterminal 1> utterance1 | ... | utteranceN
        OR
        Plan macro rules: nonterminal 2> plan
                          nonterminal(...) 2> plan

        and stores them in dictionaries.

        For sync rules, we key on (nonterminal, plan) so that we can associate all the utterances that have the same plan
        with each other.
        """

        non_macro_rules: List[Rule] = []
        macros: Dict[str, Macro] = {}

        sync_rules: DefaultDict[
            Tuple[Nonterminal, Expansion], List[Expansion]
        ] = defaultdict(list)
        utterance_rules: DefaultDict[Nonterminal, List[Expansion]] = defaultdict(list)
        schema_annotations: Dict[Nonterminal, EntitySchema] = {}

        ###
        # Step 1. Separate out macros from the rest of the rules
        ###

        parser = get_scfg_parser()
        for line in grammar_input:
            line = line.strip()

            if is_skippable(line):
                continue

            rule: Union[Rule, Macro] = parse_string(parser, line)

            if isinstance(rule, Macro):
                assert (
                    rule.name not in macros
                ), f"Macro {rule.name} cannot be defined more than once."
                macros[rule.name] = rule
            else:
                non_macro_rules.append(rule)
        ###
        # Step 2.
        # Figure out if the rule is a plan rule, an utterance rule, or a sync rule.
        # For each rule with a plan rhs, rewrite it so all macros are expanded.
        #
        # if the rule is an utterance rule or a sync rule, with tokens that are optional (e.g. x? or "the"?)
        # compile the rule into multiple rules with no optional tokens.
        ####
        for rule in non_macro_rules:
            if isinstance(rule, UtteranceRule):
                for rhs in rule.utterance_rhss:
                    utterance_rules[rule.lhs] += [
                        option for option, _ in all_possible_options(rhs)
                    ]
            elif isinstance(rule, SyncRule):
                if rule.entity_schema:
                    assert (
                        rule.lhs not in schema_annotations
                    ), "Every nonterminal can only have one schema annotation"
                    schema_annotations[rule.lhs] = rule.entity_schema

                plan_rule_no_macro = expand_macros(macros, rule.plan_rhs)
                plan_rules_no_options = all_possible_options(plan_rule_no_macro)

                ##
                # The reason for the complexity below is we need to make sure both the utterance
                # rules have the same optional nonterminals that were included/excluded from the plan.
                # Note that utterance rules can have additional optional nonterminals not present in the plan.
                ##
                for plan_rule_no_option, diff in plan_rules_no_options:
                    for utterance_rule in rule.utterance_rhss:
                        utterance_rules_no_options = [
                            option
                            for option, _ in all_possible_options(
                                modify_expansion(utterance_rule, diff)
                            )
                        ]
                        sync_rules[
                            rule.lhs, plan_rule_no_option
                        ] += utterance_rules_no_options

        return PreprocessedGrammar(sync_rules, utterance_rules, schema_annotations)

    @staticmethod
    def from_folder(folder_path: str) -> "PreprocessedGrammar":
        all_scfg_paths = find_all_scfg_paths(folder_path)
        if not all_scfg_paths:
            raise ValueError(f"No .scfg files found under {folder_path!r}.")
        all_scfg_paths.sort()
        with contextlib.ExitStack() as stack:
            line_iters = [
                iter(stack.enter_context(open(filename))) for filename in all_scfg_paths
            ]
            return PreprocessedGrammar.from_line_iter(
                itertools.chain.from_iterable(line_iters)
            )
