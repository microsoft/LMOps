# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    cast,
)

from lark import Token, Tree

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.agenda import BP, Attach, Column, Item, Meta, Predict, Scan
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import DottedRule, Grammar, Nonterm, Terminal
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.input import Position
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.keydefaultdict import KeyDefaultDict


class PackedForest(Generic[Terminal]):
    @property
    def data(self) -> str:
        # helper for converting to lark.Tree
        if isinstance(self, Ambig):
            return "_ambig"
        elif isinstance(self, Node):
            n: Node[Terminal] = cast(Node[Terminal], self)
            rule = n.rule  # pylint: disable=no-member
            return rule.alias if rule.alias is not None else rule.lhs.name
        else:
            return ""

    def to_tree(self) -> Tree:
        def go(p: PackedForest[Terminal]) -> Optional[Tree]:
            if isinstance(p, (Node, Ambig)):
                return Tree(
                    data=p.data,
                    children=[
                        t
                        for c in p.ordered_children
                        for t in [go(c)]
                        if t is not None  # filter out Leaves
                    ],
                )
            # FIXME: this only works for {String,Char}Position
            elif isinstance(p, Leaf):
                pos = p.pos
                # FIXME: this is a hacky hacky way to detect RegexTokens
                #  without having to import it
                if hasattr(pos, "string") and hasattr(pos, "i"):
                    s = cast(str, getattr(pos, "string"))
                    i = cast(int, getattr(pos, "i"))
                    ith_char = s[i : i + 1]
                    if ith_char != p.terminal:
                        # p.terminal is RegexToken
                        return Token("", ith_char)  # type: ignore
            # non-regex Leaves are ignored in lark.Trees
            return None

        return go(self)

    @staticmethod
    def plus(
        a: "PackedForest[Terminal]", b: "PackedForest[Terminal]"
    ) -> "PackedForest[Terminal]":
        if isinstance(a, Fail):
            return b
        if isinstance(b, Fail):
            return a
        if a == b:
            return a
        # attempt to collate rhs when rules match:
        by_rule = defaultdict(list)
        for c in a.flatten() + b.flatten():
            by_rule[(c.rule, len(c.children))].append(c)
        result: List[PackedForest[Terminal]] = [
            Node(
                rule=rule,
                children=[
                    PackedForest.sum(cs) for cs in (zip(*[x.children for x in xs]))
                ],
            )
            for (rule, _), xs in by_rule.items()
        ]
        if len(result) == 1:
            return result[0]
        else:
            return Ambig(children=result)

    def flatten(self) -> List["Node[Terminal]"]:
        if isinstance(self, Node):
            return [cast(Node, self)]
        elif isinstance(self, Ambig):
            a: Ambig[Terminal] = cast(Ambig, self)
            return [
                c for x in a.children for c in x.flatten()  # pylint: disable=no-member
            ]
        else:
            return []

    @staticmethod
    def sum(xs: Iterable["PackedForest[Terminal]"]) -> "PackedForest[Terminal]":
        return reduce(PackedForest.plus, xs, Fail())

    def __lt__(self, other):
        return str(self).__lt__(str(other))


@dataclass(frozen=True)
class Node(PackedForest, Generic[Terminal]):
    rule: DottedRule[Terminal]
    children: List[PackedForest[Terminal]]

    @property
    def ordered_children(self) -> List[PackedForest[Terminal]]:
        return self.children


@dataclass(frozen=True)
class Ambig(PackedForest, Generic[Terminal]):
    children: List[PackedForest[Terminal]]

    @property
    def ordered_children(self) -> List[PackedForest[Terminal]]:
        # order doesn't matter for ambig nodes, so canonicalize here
        return list(sorted(self.children))

    def __eq__(self, other):
        if not isinstance(other, Ambig):
            return False
        return self.ordered_children == other.ordered_children


@dataclass(frozen=True)
class Leaf(PackedForest, Generic[Terminal]):
    terminal: Terminal
    # we save `pos` in case `terminal` is a regex,
    # which doesn't remember which char it matched
    pos: Optional[Position[Terminal]] = None

    def __repr__(self):
        return f"Leaf({self.terminal})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Leaf):
            return False
        return self.terminal == other.terminal


@dataclass(frozen=True)
class Fail(PackedForest, Generic[Terminal]):
    pass


class EarleyChart(Generic[Terminal]):
    """A chart for Earley's algorithm.  Allows fine-grained external control,
    or it can be subclassed to add a control mechanism like exhaustive search
    or beam search.  The API is called with positions in the input."""

    # maps an input position to its column, making a new column if necessary
    _cols: KeyDefaultDict[Position[Terminal], Column[Terminal]]

    def __init__(self, grammar: Grammar[Terminal]) -> None:
        self.grammar = grammar
        self._cols = KeyDefaultDict(Column[Terminal])

    def seek(self, nonterm: Nonterm, start_pos: Position[Terminal]) -> None:
        """
        Tells the parser to seek a constituent of type nonterminal
        starting at the given position.  Normally, the nonterminal is the start
        symbol of the grammar, and the position is the start of the input.
        """
        self._predict(None, nonterm, self._cols[start_pos])

    def was_found(
        self,
        nonterm: Nonterm,
        start_pos: Position[Terminal],
        end_pos: Position[Terminal],
    ) -> bool:
        """
        Has the parser found a constituent of type `nonterm` from the
        start to the end position, as sought by the `seek` method?  This can be used
        to answer the recognition question, though not the parsing question.
        """
        items = self.completed_items(nonterm, start_pos, end_pos)
        return next(items, None) is not None

    def completed_items(
        self,
        nonterm: Nonterm,
        start_pos: Position[Terminal],
        end_pos: Position[Terminal],
    ) -> Iterator[Tuple[Item[Terminal], Meta[Terminal]]]:
        start_col = self._cols[start_pos]
        for item, weight in self._cols[end_pos].all_items():
            if item.dotted_rule.lhs == nonterm and item.dotted_rule.is_final():
                # we found a complete nonterm
                if item.start_col == start_col:
                    yield item, weight

    def _advance_with_terminal_callback(
        self,
        pos: Position[Terminal],
        callback: Callable[[Item[Terminal], Terminal, Column[Terminal]], None],
    ) -> None:
        col = self._cols[pos]
        logging.debug("")
        logging.debug("Processing items in %s", col)
        while col:  # while agenda isn't empty
            item = col.pop()  # dequeue the next unprocessed item
            if item.dotted_rule.is_final():
                # Attach this complete constituent to its previously popped customers to the left
                logging.debug("%s => ATTACH", item)
                self._attach(item, col)
            # not incompatible with being final
            for next_symbol in item.dotted_rule.next_symbols():
                if isinstance(next_symbol, Nonterm):
                    # Looking for a complete constituent to the right
                    logging.debug("%s => PREDICT", item)
                    self._predict(item, next_symbol, col)
                else:
                    # Looking for a terminal
                    logging.debug("%s => SCAN", item)
                    callback(item, next_symbol, col)

    def advance(self, pos: Position[Terminal]) -> Iterable[Position[Terminal]]:
        """
        Processes all the remaining items ending at pos (and any
        new items ending at pos that get created as a result).
        Returns the set of new (future) ending Positions for items
        that get created as a result, so that advance can be called
        on them too (in some chosen agenda order).
        """
        touched: Set[Position[Terminal]] = set()

        def callback(
            item: Item[Terminal], next_symbol: Terminal, col: Column[Terminal]
        ):
            for future_pos in self._scan(item, next_symbol, col):
                if future_pos != pos:
                    touched.add(future_pos)

        self._advance_with_terminal_callback(pos, callback)
        return touched

    def advance_only_nonterminals(
        self, pos: Position[Terminal]
    ) -> Dict[Terminal, List[Item[Terminal]]]:
        """
        Processes the chart so that all items in the column for the given `pos`
        consists of rules which look like
            A → α • a β
        rather than
            A → α • B β
            A → γ •
        where α, β, and γ are sequences of terminals and nonterminals,
        A, B are nonterminals, and a is a terminal.

        Returns the set of all terminals that can come next from this position.
        """
        terminals: Dict[Terminal, List[Item[Terminal]]] = collections.defaultdict(list)

        def callback(
            item: Item[Terminal], next_symbol: Terminal, _col: Column[Terminal]
        ):
            # Assumes that `self._scan(item, next_symbol, col)` would
            # have not touched this column, only future ones
            terminals[next_symbol].append(item)

        self._advance_with_terminal_callback(pos, callback)

        # Put back all the items which have a terminal as the next item
        self._cols[pos].unpop_using_pred(
            lambda item: bool(item.dotted_rule.rhs)
            and not isinstance(item.dotted_rule.rhs[0], Nonterm)
        )
        return terminals

    def advance_with_terminal(
        self,
        pos: Position[Terminal],
        terminal: Terminal,
        items: Iterable[Item[Terminal]],
    ) -> Iterable[Position[Terminal]]:
        """Scan from the given `pos` with the given `terminal`.

        The relevant items which need to be scanned must be provided explicitly,
        as they are not retrieved from the chart. They should be the items
        which make sense to scan with the provided terminal.
        """
        return {
            next_pos
            for item in items
            for next_pos in self._scan(item, terminal, self._cols[pos])
        }

    def _predict(
        self,
        customer: Optional[Item[Terminal]],
        nonterm: Nonterm,
        col: Column[Terminal],
    ) -> None:
        """
        Starts looking for this nonterm starting at col, to continue the given customer.
        customer can also be None if looking for nonterm is our top-level goal.
        """
        # first, see if some values of the predicted thing have *already* been found due to order violation.
        # if so, they serve this customer immediately.
        if customer is not None:
            logging.debug(
                "\tLooking right for complete %s servers starting at %s",
                nonterm,
                col.pos,
            )
            for server, future_col in col.servers[nonterm]:
                logging.debug("\tFound server %s in %s", server, future_col)
                for new_item in customer.scan(nonterm):
                    future_col.push(
                        new_item,
                        Meta.pure(Attach(server=server, customer=customer, col=col)),
                    )
                    logging.debug("\tAttached to get: %s in %s", new_item, future_col)

        # now do a traditional predict operation
        if nonterm not in col.predicted:
            # don't add all the same rules to this col if we already did
            col.predicted.add(nonterm)  # remember not to do it again
            for rule in self.grammar.expansions[nonterm]:
                new_item = Item[Terminal](rule, col)
                col.push(
                    item=new_item, meta=Meta.pure(Predict(new_item=new_item)),
                )
                logging.debug("\tPredicted: %s in %s", new_item, col)

        # customer is now waiting for anything that comes back from the prediction
        if customer is not None:
            col.customers[nonterm].append(customer)

    def _scan(
        self, item: Item[Terminal], terminal: Terminal, col: Column[Terminal]
    ) -> Iterable[Position[Terminal]]:
        """
        Try to consume the terminal from the input, creating new items.
        Yield the new positions, if any, that can be reached in this way.
        There may be duplicates.
        """
        for future_pos in col.pos.scan(terminal):  # try to advance in the input
            future_col = self._cols[future_pos]
            for new_item in item.scan(terminal):  # try to advance in the rule
                future_col.push(
                    item=new_item,
                    meta=Meta.pure(Scan(item=item, terminal=terminal, col=col)),
                )
                logging.debug("\tScanned to get: %s in %s", new_item, future_col)
            yield future_pos

    @staticmethod
    def _attach(server: Item[Terminal], col: Column[Terminal]) -> None:
        """
        The complete item can now serve its customers in previous columns, advancing
        the customers' dots to create new items in column col.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        lhs = server.dotted_rule.lhs  # nonterminal of this item
        # start position of this item = end position of item to its left
        past_col = server.start_col
        logging.debug("\tLooking left for %s customers in %s", lhs, past_col)
        for customer in past_col.customers[lhs]:
            logging.debug("\tFound customer %s in %s", customer, past_col)
            for new_item in customer.scan(lhs):
                col.push(
                    new_item,
                    meta=Meta.pure(Attach(server=server, customer=customer, col=col)),
                )
                logging.debug("\tAttached to get: %s in %s", new_item, col)
        # past_col must remember that this item came looking
        past_col.servers[lhs].append((server, col))


class EarleyLRChart(EarleyChart[Terminal]):
    """
    Exhaustive Earley's algorithm (there's no pruning of positions or items at a position).
    Advances all positions from left to right (that is, in len() order).  This is one possible
    control strategy for EarleyChart.  It can be used for input strings, tries, lattices, etc.
    """

    # start position of the input (this class only supports one)
    start_pos: Position[Terminal]

    # final positions of the input that we've been able to reach
    _final_pos_set: Optional[Set[Position[Terminal]]]

    def __init__(
        self, grammar: Grammar[Terminal], start_pos: Position[Terminal]
    ) -> None:
        """A chart for a given input."""
        super().__init__(grammar)
        self.start_pos = start_pos
        self._final_pos_set = None

    def accepting_positions(self) -> Iterable[Position[Terminal]]:
        """
        Returns the final positions in the input whose strings were
        accepted by the grammar.

        This is a generator, and will return a position as soon as it
        is known to be final.  That position's column may accumulate
        more parses as the generator continues (if the input is cyclic),
        so if you want to going to look at the column to find all the parses,
        then wait until the generator has finished and not as soon as it
        yields the position.

        TODO: The return type on this is not precise enough, because
        it doesn't ensure that the accepting positions come from the
        same subtype of Position[Terminal] that was passed to seek.
        We really want to parameterize EarleyChart on both Terminal
        and a specific subclass of Position[Terminal].  Alas, it seems that
        we can't bound the type var for the latter by Position[Terminal]
        because that's generic (and covaries with Terminal).
        """

        if self._final_pos_set is not None:
            # we've been here before, so just consult cached results
            return self._final_pos_set

        # initialize parser
        self._final_pos_set = set()
        self.seek(self.grammar.root, self.start_pos)
        # positions we still have to process, sorted by len()
        pos_agenda = [self.start_pos]

        # run parser
        while pos_agenda:
            pos = heapq.heappop(pos_agenda)
            for future_pos in self.advance(pos):
                # process pos's column and accumulate more work
                heapq.heappush(pos_agenda, future_pos)
            if (
                # is complete input
                pos.is_final()
                # has a complete parse
                and self.was_found(self.grammar.root, self.start_pos, pos)
                # not previously yielded
                and pos not in self._final_pos_set
            ):
                yield pos
                self._final_pos_set.add(pos)

    def parse(self) -> PackedForest[Terminal]:
        """Returns a packed forest representation of all possible parses."""
        # force to a list so we're sure we're done parsing
        meta = self.final_meta()
        return backtrace(meta)

    def final_meta(self) -> Meta[Terminal]:
        """The Meta (currently backpointers) representing all possible parses."""
        accepting_positions = list(self.accepting_positions())
        all_metas = [
            m
            for pos in accepting_positions
            for _, m in self.completed_items(self.grammar.root, self.start_pos, pos)
        ]
        return reduce(Meta.plus, all_metas, Meta.zero())


def backtrace(meta: Meta[Terminal]) -> PackedForest[Terminal]:
    """
    Given the Meta for a completed Item, constructs a PackedForest
    representing all the possible parses.
    A Meta contains zero or more backpointers...
    Each one is handled internally with `go_bp`, and combined with
    `PackedForest.sum` to handle ambiguity (more than 1) or failure (0).
    """

    def go_bp(bp: BP) -> PackedForest[Terminal]:
        """Backtrace a single backpointer"""
        # Beginning of a rule (includes empty productions)
        if isinstance(bp, Predict):
            return Node(rule=bp.new_item.dotted_rule, children=[])
        # a completed NT
        elif isinstance(bp, Attach):
            # the parent production, e.g. `W -> X Y . Z`
            parent_item = bp.customer
            # the right-most child of parent, e.g. `Z`
            last_child_item = bp.server
            last_child_meta = bp.col.get_meta(last_child_item)
            # backtrace through the right-most child
            last_child_node = backtrace(last_child_meta)
            parent_meta = last_child_item.start_col.get_meta(parent_item)
            # backtrace through the rest of the children (e.g. `X Y`)
            return go_init(parent_meta, last_child_node)
        # a T
        elif isinstance(bp, Scan):
            # the parent production, e.g. `W -> X Y . /z/`
            parent_item = bp.item
            last_child_node = Leaf(terminal=bp.terminal, pos=bp.col.pos)
            parent_meta = bp.col.get_meta(parent_item)
            # backtrace through the rest of the children (e.g. `X Y`)
            return go_init(parent_meta, last_child_node)
        return Fail()  # unreachable

    def go_init(
        parent_meta: Meta[Terminal], last_child_node: PackedForest[Terminal]
    ) -> PackedForest[Terminal]:
        """Backtrace through parent, then append last_child_node to the list of children"""
        init_node = backtrace(parent_meta)
        # This should never be Ambig, b/c these are all parses of parent_item,
        # so the rhs symbols will line up. In that case we always collate and
        # push Ambigs down under the children (see `PackedForest.plus`).
        assert isinstance(init_node, Node)
        # stitch it back together with the right-most child
        return Node(
            rule=init_node.rule, children=init_node.children + [last_child_node]
        )

    return PackedForest.sum(go_bp(c) for c in meta.bps)
