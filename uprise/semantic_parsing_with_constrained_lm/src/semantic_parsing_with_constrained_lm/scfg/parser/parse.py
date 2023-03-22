# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Any, List, Tuple, Union, cast

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedEOF  # type: ignore[attr-defined]

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.macro import Macro
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.rule import Rule, SyncRule, UtteranceRule
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import (
    EmptyToken,
    EntitySchema,
    MacroToken,
    NonterminalToken,
    RegexToken,
    SCFGToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion


def parse_string(parser: Lark, string: str) -> Any:
    """
    Parse a string into a Rule or an Expansion.

    We annotate the return type as Any because this can return a Rule, Macro, or Expansion, and
    it seems silly to do a cast at each call site.
    """
    try:
        return TreeToRule().transform(parser.parse(string))
    except UnexpectedEOF as e:
        raise Exception(f"Line could not be parsed: {string}") from e


class TreeToRule(Transformer):
    def get_terminal_value(self, args, default_before):
        if len(args) > 1:
            base = json.loads(args[1].value)
            return json.dumps("" + base)
        else:
            base = json.loads(args[0].value)
            return json.dumps(default_before + base)

    @v_args(inline=True)
    def start(self, arg) -> Union[Macro, Rule]:
        return arg

    @v_args(inline=True)
    def start_for_test(self, arg) -> Union[Macro, Expansion, Rule]:
        return arg

    def terminal_with_space(self, args) -> TerminalToken:
        return TerminalToken(self.get_terminal_value(args, " "), optional=False)

    def optional_terminal_with_space(self, args) -> TerminalToken:
        return TerminalToken(self.get_terminal_value(args, " "), optional=True)

    def terminal_without_space(self, args) -> TerminalToken:
        return TerminalToken(self.get_terminal_value(args, ""), optional=False)

    def optional_terminal_without_space(self, args) -> TerminalToken:
        return TerminalToken(self.get_terminal_value(args, ""), optional=True)

    @v_args(inline=True)
    def nonterminal(self, name_token) -> NonterminalToken:
        return NonterminalToken(name_token.value, optional=False)

    @v_args(inline=True)
    def optional_nonterminal(self, name_token) -> NonterminalToken:
        return NonterminalToken(name_token.value, optional=True)

    @v_args(inline=True)
    def empty(self) -> EmptyToken:
        return EmptyToken()

    def regex_with_space(self, args) -> RegexToken:
        if len(args) > 1:
            prefix = ""
            value = args[1].value
        else:
            prefix = " "
            value = args[0].value

        return RegexToken(value, optional=False, prefix=prefix)

    def regex_without_space(self, arg) -> RegexToken:
        return RegexToken(arg[0].value, optional=False, prefix="")

    def plan_expansion(self, args) -> Expansion:
        return args

    def utterance_expansion(self, args) -> Expansion:
        return args

    def utterance_expansions(self, no_macro_expansions) -> List[Expansion]:
        return no_macro_expansions

    @v_args(inline=True)
    def utterance_token(self, arg) -> SCFGToken:
        return arg

    @v_args(inline=True)
    def plan_token(self, arg) -> SCFGToken:
        return arg

    @v_args(inline=True)
    def rule(self, name_token) -> str:
        return name_token.value

    def sync_rule(self, args) -> SyncRule:
        if len(args) > 3:
            rule, entity_annotation, expansions, expansion = args
            return SyncRule(rule, expansions, expansion, entity_annotation)
        else:
            rule, expansions, expansion = args
            return SyncRule(rule, expansions, expansion, None)

    @v_args(inline=True)
    def utterance_rule(self, rule, expansions) -> UtteranceRule:
        return UtteranceRule(rule, expansions)

    def entity_annotation(self, args) -> EntitySchema:
        if len(args) > 1:
            return EntitySchema(args[1], prefix="")
        return EntitySchema(args[0], prefix=" ")

    @v_args(inline=True)
    def macro_rule(self, macro_def, expansion) -> Macro:
        return Macro(macro_def[0], macro_def[1], expansion)

    def macro_def(self, args) -> Tuple[str, Tuple[str, ...]]:
        return cast(str, args[0].value), tuple(cast(str, a.value) for a in args[1:])

    def macro_apply(self, args) -> MacroToken:
        return MacroToken(args[0].value, tuple(args[1:]))


def get_scfg_parser(start_symbol: str = "start") -> Lark:
    """
    Get a parser based on the SCFG grammar. The start rule that gets appended to the grammar
    at the end depends on whether we are testing or not. If we are testing, then we want to be
    able to parse expansions outside of rules so that in our tests, we don't have to write
    lists of tokens.
    """
    scfg_grammar_path = Path(__file__).parent / "scfg_grammar.cfg"
    scfg_grammar: str
    with open(scfg_grammar_path, "r") as cf_grammar_file:
        scfg_grammar = cf_grammar_file.read()

    # Type ignoring because mypy doesn't play well with Lark.
    return Lark(scfg_grammar, ambiguity="explicit", start=start_symbol)  # type: ignore
