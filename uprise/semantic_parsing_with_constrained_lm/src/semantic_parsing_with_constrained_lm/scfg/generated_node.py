# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import lark
from pydantic.dataclasses import dataclass

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.types import Alias


@dataclass(frozen=True)
class GeneratedNode(ABC):
    @abstractmethod
    def render(self, with_treebank: bool = True) -> str:
        pass

    @abstractmethod
    def render_tokenized(self) -> List[str]:
        pass

    @abstractmethod
    def render_topological(self, with_treebank: bool = True) -> str:
        """
        Unbeknownst to Lark, the plan right hand sides of our SCFG grammar can contain sequences of expressions.
        To denote such sequences, we use a special token '"/"':

        create 2> "[x] = do the Event" "/" "describe ask create Event with [x].attendee at [x].startTime"

        Now suppose that you have an existing set of rules like:
        create 2> "describe ask create Event" with_event_modifier
        with_event_modifier 2> "with attendees \"Brian\""

        But now you also want to handle the example above. You _could_ write a new set of rules
        create_salient 2> "[x] = do the Event" "/" "describe ask create Event" with_salient_event_modifier
        with_salient_event_modifier 2> "with [x].attendee at [x].startTime"

        But suppose you already had N create rules, and you wanted to make all of them salient. You would have to
        make N more rules. This would make the grammar extremely bloated.

        Instead, what you would like to be able to do is write a single extra rule:
        with_event_modifier 2> "[x] = do the Event" "/" "with [x].attendee at [x].startTime"

        And our system should be able to automatically determine that "[x] = do the Event" should be generated first,
        before everything else.

        render_topological is a principled way of doing this. render_topological assumes that every
        right hand side actually specifies a One-Child-Policy DAG (OCPDAG) :TM:
        where every expression is a node, and each node's parent is the expression that came before it.

        For example,
        with_event_modifier 2> "[x] = do the Event" "/" "with [x].attendee at [x].startTime"

        defines a 2-node OCPDAG: "[x] = do the Event" -> "with [x].attendee at [x].startTime"

        When a rule X is used in a rule Y, we assume that the leaf node X gets substituted
        into the node representing the expression using X.

        For example,
        create 2> "describe ask create Event" with_event_modifier with_place
        with_event_modifier 2> "[x] = do the Event" "/" "with [x].attendee at [x].startTime"
        with_place 2> "[x] = do the Place" "/" "at [x]"
        defines a 3 node OCPDAG:
        "describe ask create Event" "with [x].attendee at [x].startTime"
        "do the Event" -> "describe ask create Event" "with [x].attendee at [x].startTime" "at [x]"
        "do the Place" -> "describe ask create Event" "with [x].attendee at [x].startTime" "at [x]"

        render_topological renders this graph in a topological order, with "/" added in between nodes.
        No guarantees are made about the exact ordering other than it's topological.

        See test_generated_node.py for a more complicated example.
        """

        pass

    @abstractmethod
    def render_topological_list(self, with_treebank: bool = True) -> List[str]:
        """
        Same thing as above, but instead of joining separate expressions together with "/", we keep the
        OCPDAG in a list.
        """
        pass

    @abstractmethod
    def is_split_node(self) -> bool:
        """
        Checks if the token is the special token '"/"'
        """
        pass

    @abstractmethod
    def to_lark_tree(self) -> lark.Tree:
        """
        Returns the lark.Tree we would obtain if we generated and parsed this node.
        """
        pass


@dataclass(frozen=True)
class GeneratedTerminalNode(GeneratedNode):
    generation: str

    def render(self, with_treebank: bool = True) -> str:
        return self.generation

    def render_tokenized(self) -> List[str]:
        return [self.generation]

    def render_topological(self, with_treebank: bool = True) -> str:
        return self.generation

    def render_topological_list(self, with_treebank: bool = True) -> List[str]:
        return [self.generation]

    def is_split_node(self) -> bool:
        return self.generation == "/"

    def __str__(self):
        return "T(%s)" % self.generation

    def to_lark_tree(self) -> lark.Tree:
        raise NotImplementedError


@dataclass(frozen=True)
class GeneratedNonterminalNode(GeneratedNode):
    expansion: Tuple[GeneratedNode, ...]
    alias: Optional[Alias] = None

    def render(self, with_treebank: bool = True) -> str:
        return "".join(self.render_tokenized())

    def render_tokenized(self) -> List[str]:
        return [token for node in self.expansion for token in node.render_tokenized()]

    def render_topological(self, with_treebank: bool = True) -> str:
        return " / ".join(self.render_topological_list(with_treebank))

    def render_topological_list(self, with_treebank: bool = True) -> List[str]:
        ###
        # This block of code is a manual split on the "/" special token.
        ###
        expressions: List[List[GeneratedNode]] = []
        next_expression: List[GeneratedNode] = []
        for node in self.expansion:
            if node.is_split_node():
                expressions.append(next_expression)
                next_expression = []
            else:
                next_expression.append(node)
        expressions.append(next_expression)

        generations: List[str] = []
        for expression in expressions:
            last_render: List[str] = []
            for node in expression:
                render = node.render_topological_list(with_treebank)
                last_expression = render[-1]
                last_render.append(last_expression)
                generations += render[:-1]
            generations.append("".join(last_render))

        return generations

    def is_split_node(self) -> bool:
        return False

    def to_lark_tree(self) -> lark.Tree:
        return lark.Tree(
            self.alias,
            [
                node.to_lark_tree()
                for node in self.expansion
                if isinstance(node, GeneratedNonterminalNode)
            ],
        )
