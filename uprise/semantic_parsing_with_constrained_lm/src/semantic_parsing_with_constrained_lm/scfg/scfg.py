# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

from cached_property import cached_property
from lark import Lark

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.char_grammar import CharGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import EntitySchema, NonterminalToken
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Alias, Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar

AliasedGrammar = Dict[Nonterminal, List[Tuple[Expansion, Alias]]]


class SCFG:
    """
    Compiles an .scfg grammar into an SCFG (which consists of two LarkCFGs, and a bunch of extra metadata
    to connect them to each other).

    In particular, if we have rules
    x -> utt1, plan1
    x -> utt3, plan1
    x -> utt2, plan2

    The utterance LarkCFG will look like:
    x: utt1 -> x1
    x: utt3 -> x2
    x: utt2 -> x3
    And the plan LarkCFG will look like:
    x: plan1 -> x_x1_x2
    x: plan2 -> x_x3

    x1 and x2 are aliases that can be used to identify a specific rule for a given nonterminal.

    The following is a list of fields an SCFG has. See the code for comments about what each one is.

    self.plan_lark: Tuple[str, Lark]
    self.utterance_lark: Tuple[str, Lark]

    self.plan_grammar_keyed_by_nonterminal: DefaultDict[Nonterminal, List[Expansion] = defaultdict(list)
    self.utterance_grammar_keyed_by_nonterminal: DefaultDict[Nonterminal, List[Expansion] = defaultdict(list)

    self.plan_grammar_keyed_by_alias: Dict[Alias, Expansion] = {}
    self.utterance_grammar_keyed_by_alias: Dict[Alias, Expansion] = {}

    self.plan_nonterminal_indices_by_alias: Dict[Alias, Dict[Nonterminal, int]] = {}
    self.utterance_nonterminal_indices_by_alias: Dict[Alias, Dict[Nonterminal, int]] = {}

    self.utterance_alias_to_plan_alias: Dict[Alias, List[Alias]] = {}
    self.plan_alias_to_utterance_alias: Dict[Alias, List[Alias]] = {}

    self.schema_annotations: Dict[Nonterminal, str] = preprocessed_grammar.schema_annotations
    """

    def __init__(
        self,
        preprocessed_grammar: PreprocessedGrammar,
        start: Iterable[str] = ("start",),
    ):
        self.start: List[str] = list(start)

        ###
        # These dictionaries store the grammar in an intermediate format which we will use to convert to Lark
        # parsers.
        ###
        self.plan_grammar: AliasedGrammar = defaultdict(list)
        self.utterance_grammar: AliasedGrammar = defaultdict(list)

        ###
        # This dictionary is used to keep track of the number of times a nonterminal appears on the left hand side.
        ###
        nonterminal_count: DefaultDict[Nonterminal, int] = defaultdict(int)

        ###
        # These 4 dictionaries map nonterminals and aliases to expansions.
        ###
        self.plan_grammar_keyed_by_nonterminal: DefaultDict[
            Nonterminal, List[Expansion]
        ] = defaultdict(list)
        self.utterance_grammar_keyed_by_nonterminal: DefaultDict[
            Nonterminal, List[Expansion]
        ] = defaultdict(list)

        self.plan_grammar_keyed_by_alias: Dict[Alias, Expansion] = {}
        self.utterance_grammar_keyed_by_alias: Dict[Alias, Expansion] = {}

        ###
        # These dictionaries are used to store the order in which nonterminals appear in a rule.
        # For example, if you have a rule:
        # z -> "hi" x "my name is" y, "describe me" y "describe you" x
        # for the plan, you would want to store that x appeared first and y appeared second
        # while for the utterance you would want to store that y appeared first and x appeared second.
        # This is useful for generating from Lark parse trees which look like (z1, ((x1), (y2))), which says
        # that when you used rule z1, you expanded the first nonterminal with x1 and the second nonterminal with y2.
        # Since in our rule the ordering of x and y are reversed, having something keep track of this ordering
        # (instead of recomputing this each time) can make generation slightly more efficient.
        #
        # Note that we assume that the nth appearance of a nonterminal one one side
        # corresponds to the nth appearance of that nonterminal on the other side.
        # In order to break this assumption, we would have to change the grammar format
        # to allow for something like named capture.
        ###
        self.plan_nonterminal_indices_by_alias: Dict[
            Alias, Dict[Tuple[Nonterminal, int], int]
        ] = {}
        self.utterance_nonterminal_indices_by_alias: Dict[
            Alias, Dict[Tuple[Nonterminal, int], int]
        ] = {}

        ###
        # These dictionaries store how utterance aliases and plan map to each other.
        # For example, in the example above, we want to remember that x_x1_x2 maps to [x1, x2], and x1 maps to x_x1_x2.
        #
        # This could also be computed on-demand, but we use this to make things slightly more efficient.
        ###
        self.plan_alias_to_utterance_alias: Dict[Alias, List[Alias]] = {}
        self.utterance_alias_to_plan_alias: Dict[Alias, List[Alias]] = {}

        self.plan_nonterminal_to_aliases: DefaultDict[
            Nonterminal, List[Alias]
        ] = defaultdict(list)
        self.utterance_nonterminal_to_aliases: DefaultDict[
            Nonterminal, List[Alias]
        ] = defaultdict(list)

        ###
        # These store the names of schemas that must be output by a nonterminal. For example, the rule
        # number: Number -> "1", "1" means that the nonterminal Number must output a value
        # of type Number.
        ##
        self.schema_annotations: Dict[
            Nonterminal, EntitySchema
        ] = preprocessed_grammar.schema_annotations

        def add_utterance_rule(
            rule_nonterminal: Nonterminal, utterance_right_hand_side: Expansion
        ):
            utterance_alias = "%s_%d" % (
                rule_nonterminal,
                nonterminal_count[rule_nonterminal],
            )
            nonterminal_count[rule_nonterminal] += 1

            self.utterance_grammar[rule_nonterminal].append(
                (utterance_right_hand_side, utterance_alias)
            )
            self.utterance_grammar_keyed_by_nonterminal[rule_nonterminal].append(
                utterance_right_hand_side
            )
            self.utterance_grammar_keyed_by_alias[
                utterance_alias
            ] = utterance_right_hand_side

            self.utterance_nonterminal_indices_by_alias[
                utterance_alias
            ] = get_nonterminal_ordering(utterance_right_hand_side)
            self.utterance_nonterminal_to_aliases[rule_nonterminal].append(
                utterance_alias
            )

            return utterance_alias

        def add_plan_rule(
            rule_nonterminal: Nonterminal,
            plan_right_hand_side: Expansion,
            utt_aliases: List[str],
        ):
            plan_alias = "%s_%s" % (rule_nonterminal, "_".join(utt_aliases))

            self.plan_grammar[rule_nonterminal].append(
                (plan_right_hand_side, plan_alias)
            )
            self.plan_grammar_keyed_by_nonterminal[rule_nonterminal].append(
                plan_right_hand_side
            )
            self.plan_grammar_keyed_by_alias[plan_alias] = plan_right_hand_side

            self.plan_nonterminal_indices_by_alias[
                plan_alias
            ] = get_nonterminal_ordering(plan_right_hand_side)
            self.plan_nonterminal_to_aliases[rule_nonterminal].append(plan_alias)

            return plan_alias

        sync_rules = preprocessed_grammar.sync_rules
        utterance_rules = preprocessed_grammar.utterance_rules
        for (rule_nt, plan_rhs), utterance_rhss in sync_rules.items():
            utterance_aliases = [
                add_utterance_rule(rule_nt, utterance_rhs)
                for utterance_rhs in utterance_rhss
            ]
            pa = add_plan_rule(rule_nt, plan_rhs, utterance_aliases)

            self.plan_alias_to_utterance_alias[pa] = utterance_aliases
            for ua in utterance_aliases:
                self.utterance_alias_to_plan_alias[ua] = [pa]

        for rule_nt, utterance_rhss in utterance_rules.items():
            for utterance_rhs in utterance_rhss:
                add_utterance_rule(rule_nt, utterance_rhs)

    @cached_property
    def utterance_lark(self) -> "LarkParserAndGrammar":
        """
        We are in the process of deprecating this method in favor of
        `utterance_earley`.
        """
        return construct_lark_parser(
            self.utterance_grammar, is_plan=False, start=self.start,
        )

    @cached_property
    def plan_lark(self) -> "LarkParserAndGrammar":
        """
        We are in the process of deprecating this method in favor of
        `plan_earley`.
        """
        return construct_lark_parser(self.plan_grammar, is_plan=True, start=self.start)

    @cached_property
    def utterance_earley(self) -> CharGrammar:
        return CharGrammar.from_aliased_grammar(self.utterance_grammar)

    @cached_property
    def plan_earley(self) -> CharGrammar:
        return CharGrammar.from_aliased_grammar(self.plan_grammar)

    @staticmethod
    def from_file(grammar_input_filename: str) -> "SCFG":
        with open(grammar_input_filename, "r") as input_grammar_file_obj:
            return SCFG(PreprocessedGrammar.from_line_iter(input_grammar_file_obj))

    @staticmethod
    def from_folder(folder_path: str) -> "SCFG":
        return SCFG(PreprocessedGrammar.from_folder(folder_path))


class LarkParserAndGrammar:
    def __init__(self, grammar: str, parser: Lark):
        self.grammar = grammar
        self.parser = parser


def get_nonterminal_ordering(tokens: Expansion) -> Dict[Tuple[Nonterminal, int], int]:
    """
    Given a list of tokens, return a dictionary where the keys are nonterminals
    and the values tell you the ordering in which those nonterminals appear.

    E.g. for a list of tokens ['"hi"', 'x', '"boo"', 'y'] this method returns
    {'x': 0, 'y': 1}
    """
    nonterminal_ordering = {}
    nonterminal_count: DefaultDict[Nonterminal, int] = defaultdict(int)

    for i, token in enumerate([t for t in tokens if isinstance(t, NonterminalToken)]):
        nonterminal_ordering[(token.value, nonterminal_count[token.value])] = i
        nonterminal_count[token.value] += 1

    return nonterminal_ordering


def convert_to_lark_rule(tokens: Expansion, is_plan: bool = False) -> str:
    """
    Given a list of tokens, append "i" to the end of terminals. This indicates to Lark
    that the token is case_insensitive.
    """
    if is_plan:
        return " ".join(token.value for token in tokens)
    else:
        return " ".join(token.lark_value for token in tokens)


def construct_lark_parser(
    grammar: Dict[Nonterminal, List[Tuple[Expansion, Alias]]],
    is_plan: bool,
    **lark_kwargs,
) -> "LarkParserAndGrammar":
    lark_grammar = []
    # We sort the grammar so that it is consistently constructed in the same order for tests.
    for rule_nonterminal in sorted(grammar):
        rule_strs = (
            f"{convert_to_lark_rule(rhs, is_plan=is_plan)} -> {alias}"
            for rhs, alias in grammar[rule_nonterminal]
        )
        lark_grammar.append("%s: %s" % (rule_nonterminal, "\n|".join(rule_strs)))
    # Append "" so that the joined string terminates in a newline
    lark_grammar.append("")
    lark_grammar_str = "\n".join(lark_grammar)

    # For some reason, mypy complains about ths next line with "missing named arguments".
    return LarkParserAndGrammar(lark_grammar_str, Lark(lark_grammar_str, ambiguity="explicit", **lark_kwargs))  # type: ignore
