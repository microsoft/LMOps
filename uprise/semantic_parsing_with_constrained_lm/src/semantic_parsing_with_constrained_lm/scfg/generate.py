# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple, cast

from lark import Token, Tree

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.util import (
    IteratorGenerator,
    cross_product,
    head_or_random,
    maybe_randomize,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.char_grammar import CharGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.generated_node import (
    GeneratedNode,
    GeneratedNonterminalNode,
    GeneratedTerminalNode,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import (
    EmptyToken,
    NonterminalToken,
    RegexToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Alias, Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.string_utils import detokenize

ASCII_CHARS = [chr(i) for i in range(32, 127)]


@dataclass(frozen=True)
class SourceToTargetIndexes:
    """
    a set of indexes used to go from a source grammar to a target grammar
    """

    target_grammar_by_alias: Dict[Alias, Expansion]
    target_nonterminal_to_aliases: DefaultDict[Nonterminal, List[Alias]]
    source_nonterminal_indices_by_alias: Dict[Alias, Dict[Tuple[Nonterminal, int], int]]
    source_alias_to_target_alias: Dict[Alias, List[Alias]]

    @staticmethod
    def from_scfg(scfg: SCFG, source_is_plan: bool) -> "SourceToTargetIndexes":
        """
        Given an scfg, return a SourceToTargetIndexes where the indexes are set
        depending on whether we are going from plan to utterance or utterance to plan.
        """
        if source_is_plan:
            return SourceToTargetIndexes(
                target_grammar_by_alias=scfg.utterance_grammar_keyed_by_alias,
                target_nonterminal_to_aliases=scfg.utterance_nonterminal_to_aliases,
                source_nonterminal_indices_by_alias=scfg.plan_nonterminal_indices_by_alias,
                source_alias_to_target_alias=scfg.plan_alias_to_utterance_alias,
            )
        else:
            return SourceToTargetIndexes(
                target_grammar_by_alias=scfg.plan_grammar_keyed_by_alias,
                target_nonterminal_to_aliases=scfg.plan_nonterminal_to_aliases,
                source_nonterminal_indices_by_alias=(
                    scfg.utterance_nonterminal_indices_by_alias
                ),
                source_alias_to_target_alias=scfg.utterance_alias_to_plan_alias,
            )


def generate_from_parse_tree(
    source_parse_tree: Tree, scfg: SCFG, source_is_plan: bool, randomize: bool
) -> Iterator[GeneratedNonterminalNode]:
    """
    Given a parse tree for a source grammar, generate from a target grammar.

    source_is_plan is True if we are going from plan to utterance and false if we are going from utterance
    to plan.

    The reason we generate to the [[GeneratedNode]] intermediate representation, as opposed to directly to a string
    is so that we can generate the string separately in different ways. See the render methods in [[GeneratedNode]].
    (Currently there is only one - there will be another one in a future PR).

    If sample is set to true, we only return one possible generation.
    """
    indexes = SourceToTargetIndexes.from_scfg(scfg, source_is_plan)
    return _generate_from_parse_tree_impl(source_parse_tree, indexes, randomize)


def _generate_from_parse_tree_impl(
    source_parse_tree: Tree, indexes: SourceToTargetIndexes, randomize: bool
) -> Iterator[GeneratedNonterminalNode]:
    # TODO: make this less deeply nested and lower max-nested-blocks in pylintrc
    source_alias = source_parse_tree.data
    if source_alias == "_ambig":
        for tree in maybe_randomize(source_parse_tree.children, randomize):
            for result in _generate_from_parse_tree_impl(
                cast(Tree, tree), indexes, randomize
            ):
                yield result
    else:
        target_aliases = indexes.source_alias_to_target_alias[source_alias]
        nonterminal_indices_by_alias = indexes.source_nonterminal_indices_by_alias[
            source_alias
        ]

        for target_alias in maybe_randomize(target_aliases, randomize):
            generations_per_token: List[Iterable[GeneratedNode]] = []
            nonterminal_count: DefaultDict[Nonterminal, int] = defaultdict(int)
            rhs = indexes.target_grammar_by_alias[target_alias]
            for token in rhs:
                token_generations: Iterable[GeneratedNode]
                if isinstance(token, NonterminalToken):
                    nonterminal_number = nonterminal_count[token.value]
                    if (
                        token.value,
                        nonterminal_number,
                    ) in nonterminal_indices_by_alias:
                        nonterminal_index = nonterminal_indices_by_alias[
                            (token.value, nonterminal_number)
                        ]
                        child = source_parse_tree.children[nonterminal_index]
                        nonterminal_count[token.value] += 1

                        if isinstance(token, RegexToken):
                            token_generations = [
                                GeneratedTerminalNode(
                                    token.render_matching_value(
                                        cast(Token, child).value
                                    )
                                )
                            ]
                        else:

                            # freezes `child` so it doesn't change out from under us
                            def it(c: Tree):
                                return lambda: _generate_from_parse_tree_impl(
                                    cast(Tree, c), indexes, randomize,
                                )

                            token_generations = IteratorGenerator(it(cast(Tree, child)))
                    else:
                        ###
                        # If the nonterminal doesn't appear in the source rule, then we can ignore the
                        # parse tree and just generate randomly from the target grammar.
                        ###
                        if isinstance(token, RegexToken):
                            token_generations = _generate_from_ascii_char_regex(
                                token, randomize
                            )
                        else:
                            # freezes `token.value` so it doesn't change out from under us
                            def gen(val: str):
                                return lambda: generate_from_grammar_and_nonterminal(
                                    indexes.target_grammar_by_alias,
                                    indexes.target_nonterminal_to_aliases,
                                    randomize,
                                    val,
                                )

                            token_generations = IteratorGenerator(gen(token.value))
                elif isinstance(token, (EmptyToken, TerminalToken)):
                    token_generations = [GeneratedTerminalNode(token.render())]
                else:
                    raise Exception(f"Unexpected Token: {token}")

                generations_per_token.append(token_generations)
            for instantiated_expansion in cross_product(*generations_per_token):
                yield GeneratedNonterminalNode(instantiated_expansion, target_alias)


def _generate_from_ascii_char_regex(
    token: RegexToken, randomize: bool,
) -> Iterator[GeneratedTerminalNode]:
    """Generates all ascii characters that match the given RegexToken. """
    return (
        GeneratedTerminalNode(c)
        for c in maybe_randomize(ASCII_CHARS, randomize)
        if token.compiled.match(c)
    )


def generate_from_grammar_and_nonterminal(
    grammar_by_alias: Dict[Alias, Expansion],
    nonterminal_to_aliases: Dict[Nonterminal, List[Alias]],
    randomize: bool,
    nonterminal="start",
) -> Iterator[GeneratedNonterminalNode]:
    """
    Generate from a grammar, given a nonterminal.
    """
    alias_to_iterator: Dict[Alias, Iterator[Tuple[GeneratedNode, ...]]] = {}
    aliases = copy(nonterminal_to_aliases[nonterminal])
    while aliases:
        alias = head_or_random(aliases, randomize)
        if alias not in alias_to_iterator:
            expansion = grammar_by_alias[alias]
            generations_per_token: Tuple[Iterable[GeneratedNode], ...] = ()
            for token in expansion:
                if isinstance(token, RegexToken):
                    # TODO: EarleyPartialParse doesn't require the
                    #  space in the grammar to be a regex. So when we switch
                    #  over definitively to using them, we can delete this `if`
                    #  branch, and change the `cmspace -> !/ / , #e` rule in
                    #  smpa_fresh/fresh.scfg to be `cmspace -> !" " , #e`.
                    generations_per_token += (
                        _generate_from_ascii_char_regex(token, randomize),
                    )
                elif isinstance(token, NonterminalToken):
                    # functools.partial freezes `token`, otherwise weird
                    # python scoping would cause bugginess here
                    generations_per_token += (
                        IteratorGenerator(
                            functools.partial(
                                generate_from_grammar_and_nonterminal,
                                grammar_by_alias,
                                nonterminal_to_aliases,
                                randomize,
                                token.value,
                            )
                        ),
                    )
                elif isinstance(token, (EmptyToken, TerminalToken)):
                    generations_per_token += (
                        cast(
                            Iterable[GeneratedNode],
                            [GeneratedTerminalNode(token.render())],
                        ),
                    )
                else:
                    raise Exception(f"Unexpected Token: {token}")
            alias_to_iterator[alias] = cross_product(*generations_per_token)
        iterator = alias_to_iterator[alias]
        try:
            yield GeneratedNonterminalNode(next(iterator), alias)
        except StopIteration:
            aliases.remove(alias)


def sample_synchronously(
    scfg: SCFG, max_samples: int, nonterminal: Nonterminal = "start"
) -> Iterator[Tuple[GeneratedNonterminalNode, GeneratedNonterminalNode]]:
    indexes = SourceToTargetIndexes.from_scfg(scfg, source_is_plan=False)
    seen = set()
    num_times_sampled = 0
    # TODO: this infinite loops when you try to sample from the `crud_parse_gpt3` grammar
    # We shouldn't be trying to generate from that grammar, but it would be nicer to
    # not infinite loop (throw an error? just work?).
    while True:
        if num_times_sampled >= max_samples:
            break

        try:
            utterance_generated_node = next(
                generate_from_grammar_and_nonterminal(
                    scfg.utterance_grammar_keyed_by_alias,
                    scfg.utterance_nonterminal_to_aliases,
                    randomize=True,
                    nonterminal=nonterminal,
                )
            )
            num_times_sampled += 1
        except StopIteration:
            break
        if utterance_generated_node in seen:
            continue
        seen.add(utterance_generated_node)
        for plan_generated_node in _generate_from_parse_tree_impl(
            utterance_generated_node.to_lark_tree(), indexes, randomize=True
        ):
            yield utterance_generated_node, plan_generated_node


def generate_synchronously(
    scfg: SCFG, randomize: bool, nonterminal: Nonterminal = "start",
) -> Iterator[Tuple[GeneratedNonterminalNode, GeneratedNonterminalNode]]:
    indexes = SourceToTargetIndexes.from_scfg(scfg, source_is_plan=False)
    for utterance_generated_node in generate_from_grammar_and_nonterminal(
        scfg.utterance_grammar_keyed_by_alias,
        scfg.utterance_nonterminal_to_aliases,
        randomize,
        nonterminal,
    ):
        tree = utterance_generated_node.to_lark_tree()
        for plan_generated_node in _generate_from_parse_tree_impl(
            tree, indexes, randomize
        ):
            yield utterance_generated_node, plan_generated_node


def parse_and_render(
    scfg: SCFG, s: str, source_is_plan: bool, max_depth: Optional[int] = None
) -> Iterator[str]:
    """
    :param scfg: The grammar to parse with
    :param s: The string you want to parse
    :param source_is_plan: Whether the string is a plan or an utterance
    :param max_depth: When non-None, only parses with depth less than or equal
     to `max_depth` will be returned.
    :return: If the string is a plan, return the utterance. If the string is an utternace,
    return the plan.
    """

    grammar: CharGrammar = scfg.plan_earley if source_is_plan else scfg.utterance_earley
    parses: Iterator[Tree] = grammar.parses(s, max_depth=max_depth)
    for tree in parses:
        for g in generate_from_parse_tree(
            tree, scfg, source_is_plan=source_is_plan, randomize=False
        ):
            yield g.render_topological(with_treebank=False)


def compute_branches(
    grammar_by_alias: Dict[Alias, Expansion],
    nonterminal_to_aliases: Dict[Nonterminal, List[Alias]],
    generation_so_far=None,
    nonterminals_expanded=None,
    nonterminals_to_expand=None,
    depth=0,
    max_depth=10,
    branches: DefaultDict[int, List] = None,
):
    """
    A CF grammar induces a tree, where the structure of the tree represents all possible left-to-right expansions
    of the start rule. Every node the expansion of a nonterminal.

    For example: given the rules:
    start -> create
    start -> update

    create -> "Set up time" with_person with_subject
    with_person -> "with Bob"
    with_subject -> "called BobTime"

    We can induce a tree where the root node is `start` and it has 2 children representing its 2 different rules
    Then the `create` node has 1 child, the `with_person` node. Since `with_person` has no further nonterminals,
    the child of the `with_person` node is the `with_subject` node.

    This method returns [[branches]], which maps depth -> a list of tuples representing each node at that depth.
    Each tuple contains the generation so far, the list of nonterminals that have been expanded so far, the list of
    nonterminals that still need to be expanded, and the number of children.
    """

    if generation_so_far is None:
        generation_so_far = []
    if nonterminals_expanded is None:
        nonterminals_expanded = []
    if nonterminals_to_expand is None:
        nonterminals_to_expand = [NonterminalToken("start", False)]
    if branches is None:
        branches = defaultdict(list)

    if depth >= max_depth or len(nonterminals_to_expand) == 0:
        return branches

    nonterminal = None
    next_index = None
    next_generation_so_far = copy(generation_so_far)
    for i, token in enumerate(nonterminals_to_expand):
        if isinstance(token, NonterminalToken):
            nonterminal = token.value
            next_index = i
            break
        if isinstance(token, (EmptyToken, TerminalToken)):
            next_generation_so_far.append(token.render())
        else:
            raise Exception(f"Unexpected Token: {token}")

    if next_index is None:
        # This means that there are no more nonterminals left to expand.
        next_index = len(nonterminals_to_expand)

    aliases = nonterminal_to_aliases[nonterminal]

    next_nonterminals_expanded = copy(nonterminals_expanded)
    next_nonterminals_expanded.append(nonterminal)
    branches[depth].append(
        (
            next_generation_so_far,
            next_nonterminals_expanded,
            nonterminals_to_expand[next_index:],
            len(aliases),
        )
    )
    for alias in aliases:
        expansion = grammar_by_alias[alias]
        compute_branches(
            grammar_by_alias,
            nonterminal_to_aliases,
            next_generation_so_far,
            next_nonterminals_expanded,
            list(expansion) + nonterminals_to_expand[next_index + 1 :],
            depth + 1,
            max_depth,
            branches,
        )

    return branches


def expand_rest(
    grammar_by_alias, nonterminal_to_aliases, generation_so_far, nonterminals_to_expand
):
    complete_generation = copy(generation_so_far)
    for token in nonterminals_to_expand:
        if isinstance(token, NonterminalToken):
            complete_generation.append(
                next(
                    generate_from_grammar_and_nonterminal(
                        grammar_by_alias,
                        nonterminal_to_aliases,
                        randomize=True,
                        nonterminal=token.value,
                    )
                ).render()
            )
        elif isinstance(token, (EmptyToken, TerminalToken)):
            complete_generation.append(token.render())
        else:
            raise Exception(f"Unexpected Token: {token}")

    return detokenize(complete_generation)
