# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains PartialParse implementations for use with Earley grammars.

Currently there are 2 implementations:
- EarleyPartialParse
  - Lazily builds a trie of all utterances allowed by the grammar,
    using `advance_only_nonterminals` and `advance_with_terminal`.
  - Takes advantage of `ordered_ids` and `top_k` to stop work once enough valid
    tokens have been found.
  - Regexes must cover contiguous spans from the input utterance.
- UTF8EarleyPartialParse
  - Like EarleyPartialParse, but uses UTF-8 encoded byte strings as the
    underlying.
"""
import collections
import dataclasses
import itertools
from dataclasses import dataclass
from typing import (
    Any,
    AnyStr,
    ClassVar,
    DefaultDict,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import torch
from cached_property import cached_property
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.agenda import Item
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.earley import EarleyChart
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.grammar import Grammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.earley.input import Position
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.char_grammar import Char
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.earley_grammar import CFTerminal, EarleyCFGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.parser.token import RegexToken
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.search import PartialParse


@dataclass(frozen=True)
class CFPosition(Position[CFTerminal]):
    content: Tuple[CFTerminal, ...] = ()

    _prev: Optional["CFPosition"] = dataclasses.field(
        default=None, compare=False, repr=False
    )
    _last: Optional["CFTerminal"] = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def scan(self, terminal: CFTerminal) -> Iterable["CFPosition"]:
        if (
            isinstance(terminal, str)
            and self.content
            and isinstance(self.content[-1], str)
        ):
            return (
                CFPosition(
                    self.content[:-1] + (self.content[-1] + terminal,),
                    _prev=self,
                    _last=terminal,
                ),
            )
        else:
            return (CFPosition(self.content + (terminal,), _prev=self, _last=terminal),)

    def is_final(self) -> bool:
        """Is this a final position in the input, i.e., are we allowed to end here?"""
        return True

    def __len__(self) -> int:
        """
        A measurement of how far along the position is in the input (e.g., the prefix length).
        May be used by some control strategies to process positions in left-to-right order.
        """
        return sum(len(x) if isinstance(x, str) else 1 for x in self.content)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CFPosition) and self.content == other.content

    def __hash__(self) -> int:
        return hash(self.content)


@dataclass(frozen=True)
class CFPositionWithCopyInfo:
    pos: Position[CFTerminal]
    copy_offsets: Optional[FrozenSet[int]] = None
    # TODO: Put this field in a separate class.
    partial_utf8_bytes: Optional[bytes] = None


@dataclass
class GrammarTokenizerInfo:
    grammar: Grammar[CFTerminal]
    id_to_token: Dict[int, str]
    id_to_utf8_token: Dict[int, bytes]
    start_position: CFPosition = CFPosition()

    @classmethod
    def create(
        cls,
        tokenizer: GPT2Tokenizer,
        preprocessed_grammar: PreprocessedGrammar,
        for_plans: bool,
    ) -> "GrammarTokenizerInfo":
        grammar = cls._preproc_to_grammar(preprocessed_grammar, for_plans)

        id_to_token: Dict[int, str] = {}
        id_to_utf8_token: Dict[int, bytes] = {}
        # encoded_token: UTF-8 encoded strings where bytes corresponding to
        # control characters in ASCII have been mapped to other characters
        for encoded_token, token_id in tokenizer.encoder.items():
            # token_bytes: UTF-8 encoded string
            token_bytes = bytes(tokenizer.byte_decoder[c] for c in encoded_token)
            id_to_utf8_token[token_id] = token_bytes
            try:
                token = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Sometimes token_bytes is cut off inside a UTF-8 byte sequence
                # for a single Unicode character.
                # We just discard those tokens for `token_trie` and `id_to_token`.
                continue
            id_to_token[token_id] = token

        return GrammarTokenizerInfo(grammar, id_to_token, id_to_utf8_token)

    @classmethod
    def from_tokens_list(
        cls,
        tokens: List[str],
        preprocessed_grammar: PreprocessedGrammar,
        for_plans: bool,
    ) -> "GrammarTokenizerInfo":
        grammar = cls._preproc_to_grammar(preprocessed_grammar, for_plans)
        id_to_token = dict(enumerate(tokens))
        id_to_utf8_token = {i: token.encode("utf-8") for i, token in enumerate(tokens)}
        return GrammarTokenizerInfo(grammar, id_to_token, id_to_utf8_token)

    @classmethod
    def from_utf8_tokens_list(
        cls,
        utf8_tokens: List[bytes],
        preprocessed_grammar: PreprocessedGrammar,
        for_plans: bool,
    ) -> "GrammarTokenizerInfo":
        grammar = cls._preproc_to_grammar(preprocessed_grammar, for_plans)
        id_to_token: Dict[int, str] = {}
        for i, utf8_token in enumerate(utf8_tokens):
            try:
                token = utf8_token.decode("utf-8")
            except UnicodeDecodeError:
                continue
            id_to_token[i] = token
        id_to_utf8_token = dict(enumerate(utf8_tokens))
        return GrammarTokenizerInfo(grammar, id_to_token, id_to_utf8_token)

    @classmethod
    def _preproc_to_grammar(
        cls, preprocessed_grammar: PreprocessedGrammar, for_plans: bool,
    ) -> EarleyCFGrammar:
        if for_plans:
            rules = preprocessed_grammar.all_plan_rules
        else:
            rules = preprocessed_grammar.all_utterance_rules
        grammar = EarleyCFGrammar.from_preprocessed_rules(rules)
        return grammar


@dataclass
class GrammarNodeInfo(Generic[AnyStr]):
    chart: EarleyChart[CFTerminal] = dataclasses.field(repr=False)
    start_position: Position[CFTerminal] = dataclasses.field(repr=False)
    input_utterance: AnyStr
    # Only used when input_utterance is bytes.
    # TODO: Put this field in a separate class?
    initial_copy_offsets: Optional[FrozenSet[int]] = None

    def __post_init__(self):
        if not isinstance(self.input_utterance, bytes):
            return
        initial_copy_offsets = set()
        for i, byte in enumerate(self.input_utterance):
            if byte & 0b10000000 == 0 or byte & 0b11000000 == 0b11000000:
                initial_copy_offsets.add(i)
        self.initial_copy_offsets = frozenset(initial_copy_offsets)


LGN = TypeVar("LGN", bound="LazyGrammarNodeBase")
AnyChar = TypeVar("AnyChar", Char, int)


@dataclass
class LazyGrammarNodeBase(Generic[AnyStr, AnyChar]):
    """A lazily constructed trie of all possible valid strings in the grammar."""

    info: GrammarNodeInfo[AnyStr]
    depth: int

    # Stores information about all leaves of the trie rooted at this node, in a flattened way.
    descendants: DefaultDict[
        Tuple[int, AnyStr],
        DefaultDict[Tuple[CFPositionWithCopyInfo, CFTerminal], List[Item[CFTerminal]]],
    ] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
    )
    # Stores all regex terminals that could occur next after this node,
    # and the Items that we need to advance with if we scan that regex.
    regexes: Optional[
        DefaultDict[Tuple[CFPositionWithCopyInfo, RegexToken], List[Item[CFTerminal]]]
    ] = None

    # Stores all terminals that are completed at this node, and the Items that
    # we need to advance with to determine what can come next according to the
    # grammar.
    finished_terminals: Optional[
        DefaultDict[Tuple[CFPositionWithCopyInfo, CFTerminal], List[Item[CFTerminal]]]
    ] = None

    @cached_property
    def _next_pos_set(self) -> Set[CFPositionWithCopyInfo]:
        if not self.finished_terminals:
            return set()

        result = set()
        for (pos_with_copy_info, terminal), items in self.finished_terminals.items():
            [next_pos] = self.info.chart.advance_with_terminal(
                pos_with_copy_info.pos, terminal, items
            )
            if isinstance(terminal, RegexToken):
                result.add(
                    CFPositionWithCopyInfo(
                        next_pos,
                        pos_with_copy_info.copy_offsets,
                        pos_with_copy_info.partial_utf8_bytes,
                    )
                )
            else:
                result.add(CFPositionWithCopyInfo(next_pos,))
        return result

    def process_terminal(self, t: str) -> AnyStr:
        raise NotImplementedError

    @property
    def regex_children_type(self) -> Type["LazyGrammarNodeRegexChildrenBase"]:
        raise NotImplementedError

    @cached_property
    def children(self: LGN) -> Mapping[AnyChar, LGN]:
        """Used for advancing one character in the trie and reaching the next node."""
        next_depth = self.depth + 1

        # If we're at the end of a terminal, we need to advance further to determine new children.
        for next_pos in self._next_pos_set:
            for next_terminal, next_items in self.info.chart.advance_only_nonterminals(
                next_pos.pos
            ).items():
                if isinstance(next_terminal, str):
                    # TODO: Drop copy_offsets from next_pos?
                    self.descendants[self.depth, self.process_terminal(next_terminal)][
                        next_pos, next_terminal
                    ].extend(next_items)
                elif isinstance(next_terminal, RegexToken):
                    if self.regexes is None:
                        self.regexes = collections.defaultdict(list)
                    self.regexes[next_pos, next_terminal].extend(next_items)
                else:
                    raise ValueError(next_terminal)

        result: Dict[AnyChar, LGN] = {}
        # TODO: Do less work when len(self.descendants) is 1
        for (num_prev_chars, terminal), scan_infos in self.descendants.items():
            next_node = result.setdefault(
                terminal[self.depth - num_prev_chars],
                type(self)(self.info, next_depth,),
            )
            if num_prev_chars + len(terminal) == next_depth:
                if next_node.finished_terminals:
                    for to_scan, items in scan_infos.items():
                        assert to_scan not in next_node.finished_terminals
                        next_node.finished_terminals[to_scan] = items
                else:
                    next_node.finished_terminals = scan_infos
            else:
                assert (num_prev_chars, terminal) not in next_node.descendants
                next_node.descendants[num_prev_chars, terminal] = scan_infos

        if self.regexes:
            return self.regex_children_type(result, self.regexes, self.info, next_depth)
        else:
            return result

    @cached_property
    def can_end(self) -> bool:
        return any(
            self.info.chart.was_found(
                self.info.chart.grammar.root, self.info.start_position, pos.pos
            )
            for pos in self._next_pos_set
        )


@dataclass
class LazyGrammarNodeRegexChildrenBase(Mapping[AnyChar, LGN]):
    """Used for LazyGrammarNode.children when regexes are involved."""

    underlying: Dict[AnyChar, LGN]
    regexes: Dict[Tuple[CFPositionWithCopyInfo, RegexToken], List[Item[CFTerminal]]]

    info: GrammarNodeInfo
    next_depth: int

    _regex_processed: Set[AnyChar] = dataclasses.field(default_factory=set)

    def __getitem__(self, c: AnyChar) -> LGN:
        if c not in self._regex_processed:
            for (pos_with_copy_info, regex_token), chart_items in self.regexes.items():
                # Check that we're copying a contiguous span of the input utterance
                copy_check_indices: Iterable[int]
                if pos_with_copy_info.copy_offsets is None:
                    if self.info.initial_copy_offsets is None:
                        copy_check_indices = range(len(self.info.input_utterance))
                    else:
                        copy_check_indices = self.info.initial_copy_offsets
                else:
                    copy_check_indices = pos_with_copy_info.copy_offsets
                filtered_locs = frozenset(
                    i + 1
                    for i in copy_check_indices
                    if i < len(self.info.input_utterance)
                    and self.info.input_utterance[i] == c
                )
                if not filtered_locs:
                    continue

                new_position = self._check_against_regex(
                    regex_token, c, pos_with_copy_info, filtered_locs
                )
                if new_position is None:
                    continue

                node = self.underlying.setdefault(
                    c, self.lazy_grammar_node_type(self.info, self.next_depth,),
                )
                if node.finished_terminals is None:
                    node.finished_terminals = collections.defaultdict(list)
                node.finished_terminals[new_position, regex_token,].extend(chart_items)
            self._regex_processed.add(c)

        return self.underlying[c]

    @property
    def lazy_grammar_node_type(self) -> Type[LGN]:
        raise NotImplementedError

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: AnyChar,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[AnyChar]:
        return iter(self.underlying)

    def __len__(self) -> int:
        return len(self.underlying)


class LazyGrammarNode(LazyGrammarNodeBase[str, Char]):
    """Trie where single Unicode characters are edges."""

    @property
    def regex_children_type(self) -> Type[LazyGrammarNodeRegexChildrenBase]:
        return LazyGrammarNodeRegexChildren

    def process_terminal(self, t: str) -> str:
        return t


class LazyGrammarNodeRegexChildren(
    LazyGrammarNodeRegexChildrenBase[Char, LazyGrammarNode]
):
    @property
    def lazy_grammar_node_type(self) -> Type[LazyGrammarNode]:
        return LazyGrammarNode

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: Char,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        if regex_token.compiled.match(c):
            return CFPositionWithCopyInfo(pos_with_copy_info.pos, filtered_locs)
        return None


class UTF8LazyGrammarNode(LazyGrammarNodeBase[bytes, int]):
    """Trie where bytes (from UTF-8-encoded terminals) are edges."""

    @property
    def regex_children_type(self) -> Type[LazyGrammarNodeRegexChildrenBase]:
        return UTF8LazyGrammarNodeRegexChildren

    def process_terminal(self, t: str) -> bytes:
        return t.encode("utf-8")


class UTF8LazyGrammarNodeRegexChildren(
    LazyGrammarNodeRegexChildrenBase[int, UTF8LazyGrammarNode]
):
    @property
    def lazy_grammar_node_type(self) -> Type[UTF8LazyGrammarNode]:
        return UTF8LazyGrammarNode

    def _check_against_regex(
        self,
        regex_token: RegexToken,
        c: int,
        pos_with_copy_info: CFPositionWithCopyInfo,
        filtered_locs: FrozenSet[int],
    ) -> Optional[CFPositionWithCopyInfo]:
        # Figure out what the character actually is
        # TODO: Forbid discontinuing a regex sequence in the middle of an unfinished character
        if pos_with_copy_info.partial_utf8_bytes is None:
            utf8_bytes = bytes([c])
        else:
            utf8_bytes = pos_with_copy_info.partial_utf8_bytes + bytes([c])
        try:
            unicode_char = utf8_bytes.decode("utf-8")
        except UnicodeDecodeError:
            unicode_char = None

        if unicode_char is not None:
            # If the Unicode character is complete, then check it against the regex.
            # TODO: Think about if there are ways to avoid dead ends by checking
            # bytes before they have completed a character?
            utf8_bytes = None
            if not regex_token.compiled.match(unicode_char):
                return None
        return CFPositionWithCopyInfo(pos_with_copy_info.pos, filtered_locs, utf8_bytes)


EPP = TypeVar("EPP", bound="EarleyPartialParseBase")


@dataclass
class EarleyPartialParseBase(Generic[AnyStr, LGN], PartialParse):
    info: GrammarTokenizerInfo
    grammar_node: LGN
    _next_node_cache: Dict[int, LGN] = dataclasses.field(default_factory=dict)

    _lazy_grammar_node_type: ClassVar[Type[LGN]]

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> AnyStr:
        raise NotImplementedError

    def _id_to_token(self, i: int) -> Optional[AnyStr]:
        raise NotImplementedError

    @classmethod
    def initial(cls, info: GrammarTokenizerInfo, input_utterance: str):
        chart = EarleyChart(info.grammar)
        chart.seek(info.grammar.root, info.start_position)

        grammar_node = cls._lazy_grammar_node_type(
            info=GrammarNodeInfo(
                chart,
                info.start_position,
                cls._process_input_utterance(input_utterance),
            ),
            depth=0,
        )

        start = CFPositionWithCopyInfo(info.start_position)
        for next_terminal, next_items in chart.advance_only_nonterminals(
            info.start_position
        ).items():
            if isinstance(next_terminal, str):
                grammar_node.descendants[
                    0, grammar_node.process_terminal(next_terminal)
                ][start, next_terminal].extend(next_items)
            elif isinstance(next_terminal, RegexToken):
                grammar_node.regexes = collections.defaultdict(list)
                grammar_node.regexes[
                    CFPositionWithCopyInfo(info.start_position), next_terminal
                ].extend(next_items)
            else:
                raise ValueError(next_terminal)
        return cls(info, grammar_node)

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        assert ordered_ids is not None
        ordered_ids_list = ordered_ids.tolist()

        def process_token_id(i: int) -> bool:
            token_str = self._id_to_token(i)
            if token_str is None:
                return False
            grammar_node = self.grammar_node

            valid_token = True
            for token_char in token_str:
                # Advance the grammar terminal trie
                # TODO: Skip forward multiple characters if possible
                grammar_node = grammar_node.children.get(token_char)
                if grammar_node is None:
                    valid_token = False
                    break

            self._next_node_cache[i] = grammar_node
            return valid_token

        def produce_valid_tokens() -> Iterator[int]:
            for i in ordered_ids_list:
                is_valid = process_token_id(i)
                if is_valid:
                    yield i

        tokens_list = list(itertools.islice(produce_valid_tokens(), top_k))
        return (
            torch.tensor(tokens_list, dtype=torch.long),
            self.grammar_node.can_end,
        )

    def append(self: EPP, token: int) -> EPP:
        grammar_node = self._next_node_cache.get(token)
        if grammar_node is None:
            grammar_node = self.grammar_node
            for char in self.info.id_to_token[token]:
                grammar_node = grammar_node.children[char]
        return type(self)(self.info, grammar_node)


class EarleyPartialParse(EarleyPartialParseBase[str, LazyGrammarNode]):
    _lazy_grammar_node_type: ClassVar[Type[LazyGrammarNode]] = LazyGrammarNode

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> str:
        return input_utterance

    def _id_to_token(self, i: int) -> str:
        return self.info.id_to_token.get(i)


class UTF8EarleyPartialParse(EarleyPartialParseBase[bytes, UTF8LazyGrammarNode]):
    _lazy_grammar_node_type: ClassVar[Type[UTF8LazyGrammarNode]] = UTF8LazyGrammarNode

    @classmethod
    def _process_input_utterance(cls, input_utterance: str) -> bytes:
        return input_utterance.encode("utf-8")

    def _id_to_token(self, i: int) -> bytes:
        return self.info.id_to_utf8_token[i]
