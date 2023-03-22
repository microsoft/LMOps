# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import dataclasses
import gc
import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Awaitable,
    Callable,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import HS, AutoregressiveModel


class PartialParse(ABC):
    @abstractmethod
    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Returns possible ways to extend the current prefix.

        The Tensor is of type long and 1-dimensional, with no duplicate values,
        containing the IDs of the tokens that we could append.
        If it is None, then any token is allowed.
        The bool indicates whether we are allowed to terminate here.

        If ordered_ids and top_k are not None, this may optionally return only
        the first `top_k` token IDs from ordered_ids which comports with the
        grammar, instead of all such token IDs.
        """
        pass

    @abstractmethod
    def append(self, token: int) -> "PartialParse":
        """Return a new PartialParse created by appending this token."""
        pass


class NullPartialParse(PartialParse):
    """PartialParse which admits any sequence."""

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        return None, True

    def append(self, token: int) -> "PartialParse":
        return self


class StartsWithSpacePartialParse(PartialParse):
    def __init__(self, tokenizer: GPT2Tokenizer):
        valid_tokens = []
        for encoded_token, token_id in tokenizer.encoder.items():
            if tokenizer.byte_decoder[encoded_token[0]] == 32:
                valid_tokens.append(token_id)
        self.valid_tokens = torch.tensor(valid_tokens)

    def allowed_next(
        self, ordered_ids: Optional[torch.Tensor] = None, top_k: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        return self.valid_tokens, False

    def append(self, token: int) -> "PartialParse":
        return NullPartialParse()


PSNSub = TypeVar("PSNSub", bound="PackedSearchNode")


# https://github.com/python/mypy/issues/5374
@dataclass(frozen=True, eq=True)  # type: ignore
class PackedSearchNode(ABC):
    """Contains all state for SearchNode in compact form.

    ConstrainedDecodingProblem contains a cache for `expand`, with PackedSearchNode as the key.
    In order to start beam search from some arbitrary state, clients can construct PackedSearchNode cheaply
    (i.e. without actually running a neural network, or creating a PartialParse object).
    Then, if the PackedSearchNode is in the cache already, then we can look up its expansion cheaply.
    If it's not in the cache, `ConstrainedDecodingProblem.unpacker` can turn it into a SearchNode.
    """

    # Output tokens generated so far.
    tokens: Tuple[int, ...]

    @abstractmethod
    def append(self: PSNSub, token: int) -> PSNSub:
        pass


@dataclass
class FullSearchNode(Generic[HS]):
    packed: PackedSearchNode

    partial_parse: PartialParse
    hidden_state: Optional[HS] = dataclasses.field(repr=False)
    # Stores information about how to expand this SearchNode.
    # It's computed lazily because we will likely never actually expand any
    # particular SearchNode.
    decoding_output: "Optional[DecodingOutput]" = None

    is_finished: bool = False
    cost: float = 0
    unnormalized_cost: float = 0

    @property
    def tokens(self) -> Tuple[int, ...]:
        return self.packed.tokens


@dataclass
class DecodingOutput:
    next_logprobs: torch.Tensor
    allowed_next: Optional[torch.Tensor]
    can_end: bool


SearchNode = Union[FullSearchNode[HS], PSNSub]


@dataclass
class ConstrainedDecodingProblem(Generic[HS, PSNSub]):
    model: AutoregressiveModel[HS]
    # This function knows how to expand PackedSearchNodes.
    unpacker: Callable[
        [PSNSub], Awaitable[Tuple[PartialParse, HS, DecodingOutput, Sequence[float]]]
    ]

    # A 1D Long tensor containing the IDs of tokens indicating the end.
    eos: torch.Tensor
    length_normalization: float
    # Only use the top_k most likely next tokens in `expand`.
    # This can be set to the beam size.
    top_k: Optional[int] = None

    cache: Optional[MutableMapping[PackedSearchNode, List[FullSearchNode[HS]]]] = None

    def length_normalized(self, unnormalized: float, length: int) -> float:
        """
        Eq 14 from https://arxiv.org/abs/1609.08144, but missing the coverage term.
        TODO switch to something similar since this is probably overtuned for MT.
        :param unnormalized: log prob
        :param length: |Y|, or length of sequence.
        :return:
        """
        alpha = self.length_normalization
        lp = (5 + length) ** alpha / (5 + 1) ** alpha
        return unnormalized / lp

    async def expand(
        self, maybe_packed_node: SearchNode[HS, PSNSub],
    ) -> List[FullSearchNode[HS]]:
        if self.cache is not None:
            if isinstance(maybe_packed_node, FullSearchNode):
                packed_node = maybe_packed_node.packed
            else:
                packed_node = maybe_packed_node
            existing = self.cache.get(packed_node)
            if existing is not None:
                logging.debug("\N{DIRECT HIT} %s", packed_node)
                return existing
            else:
                logging.debug("\N{HOURGLASS WITH FLOWING SAND} %s", packed_node)

        node: FullSearchNode[HS]
        if isinstance(maybe_packed_node, FullSearchNode):
            node = maybe_packed_node
        else:
            (
                partial_parse,
                hidden_state,
                decoding_output,
                existing_logprobs,
            ) = await self.unpacker(  # type: ignore
                maybe_packed_node
            )
            unnormalized_cost = -sum(existing_logprobs)
            normalized_cost = self.length_normalized(
                unnormalized_cost, len(existing_logprobs)
            )

            node = FullSearchNode(
                maybe_packed_node,
                partial_parse,
                hidden_state,
                decoding_output,
                is_finished=False,
                cost=normalized_cost,
                unnormalized_cost=unnormalized_cost,
            )

        assert not node.is_finished

        if node.decoding_output is None:
            logprobs, new_hidden_state = await self.model.extend(
                node.tokens[-1:], node.hidden_state,
            )
            # Remove the sequence dimension
            next_logprobs = logprobs[0]

            # Remove -inf entries
            mask = next_logprobs != -float("inf")
            ordered_tokens = torch.argsort(next_logprobs, descending=True)
            allowed_next, can_end = node.partial_parse.allowed_next(
                ordered_tokens[mask[ordered_tokens]], self.top_k,
            )
        else:
            next_logprobs = node.decoding_output.next_logprobs
            new_hidden_state = node.hidden_state
            allowed_next = node.decoding_output.allowed_next
            can_end = node.decoding_output.can_end

        result: List[FullSearchNode[HS]] = []
        if can_end:
            eos_logprob = torch.logsumexp(next_logprobs[self.eos], dim=0)
            new_unnorm_cost = node.unnormalized_cost - eos_logprob.item()
            result.append(
                FullSearchNode(
                    node.packed,
                    node.partial_parse,
                    hidden_state=None,
                    is_finished=True,
                    cost=self.length_normalized(new_unnorm_cost, len(node.tokens) + 1),
                    unnormalized_cost=new_unnorm_cost,
                )
            )
        token_and_logprob_iter: Iterator[Tuple[int, torch.Tensor]]
        if allowed_next is None:
            indices = torch.arange(next_logprobs.shape[0])
            eligible_logprobs = next_logprobs
        else:
            indices = allowed_next
            eligible_logprobs = next_logprobs[allowed_next]

        if self.top_k is None:
            token_and_logprob_iter = (
                # .item() converts 0D tensor to a Python number
                (token_id_tensor.item(), logprob)
                for token_id_tensor, logprob in zip(indices, eligible_logprobs)
            )
        else:
            topk_eligible_logprobs = torch.topk(
                eligible_logprobs,
                k=min(self.top_k, eligible_logprobs.shape[0]),
                sorted=False,
            )
            token_and_logprob_iter = (
                (token_id_tensor.item(), logprob)
                for token_id_tensor, logprob in zip(
                    indices[topk_eligible_logprobs.indices],
                    topk_eligible_logprobs.values,
                )
            )

        for token, logprob in token_and_logprob_iter:
            if token in self.eos:
                continue
            new_unnorm_cost = node.unnormalized_cost - logprob.item()
            result.append(
                FullSearchNode(
                    node.packed.append(token),
                    node.partial_parse.append(token),
                    new_hidden_state,
                    cost=self.length_normalized(new_unnorm_cost, len(node.tokens) + 1),
                    unnormalized_cost=new_unnorm_cost,
                )
            )

        if self.cache is not None:
            self.cache[node.packed] = result
        return result


class BeamSearchEventListener:
    def received_expansions(
        self, node: SearchNode, expansions: List[FullSearchNode],
    ) -> None:
        pass


@dataclass
class LoggingEventListener(BeamSearchEventListener):
    gpt2_tokenizer: GPT2Tokenizer
    beam_size: int
    last_depth: Optional[int] = None

    def received_expansions(
        self, node: SearchNode, expansions: List[FullSearchNode],
    ) -> None:
        depth = len(node.tokens)
        if self.last_depth != depth:
            print(f"===== DEPTH {depth} =====")
            self.last_depth = depth

        if isinstance(node, FullSearchNode):
            node_cost_str = f" [{node.cost:.3f}]"
        else:
            node_cost_str = ""

        # `node` expands to the nodes in `expansions`.
        print(
            f"Completions for {self.gpt2_tokenizer.decode(node.tokens, clean_up_tokenization_spaces=False)!r}{node_cost_str}:"
        )
        for expansion in heapq.nsmallest(
            self.beam_size * 2, expansions, key=lambda n: n.cost
        ):
            if expansion.is_finished:
                print(f"- Finished -> [{expansion.cost:.3f}]")
            else:
                last_token = self.gpt2_tokenizer.decode(
                    expansion.tokens[-1:], clean_up_tokenization_spaces=False
                )
                if isinstance(node, FullSearchNode):
                    print(
                        f"- {last_token!r} [{expansion.unnormalized_cost - node.unnormalized_cost:.3f}] -> [{expansion.cost:.3f}]"
                    )
                else:
                    print(f"- {last_token!r} -> [{expansion.cost:.3f}]")
        if len(expansions) > self.beam_size * 2:
            print(f"... and {len(expansions) - self.beam_size * 2} more")


MAX_STEPS = 1000


async def beam_search(
    problem: ConstrainedDecodingProblem[HS, PSNSub],
    initial: SearchNode[HS, PSNSub],
    beam_size: int,
    max_steps: Optional[int] = None,
    event_listener: BeamSearchEventListener = BeamSearchEventListener(),
    keep_finished_nodes: bool = False,
) -> List[FullSearchNode]:
    max_steps = MAX_STEPS if max_steps is None else max_steps

    finished: List[FullSearchNode[HS]] = []
    finished_extra: List[FullSearchNode[HS]] = []
    beam: List[SearchNode[HS, PSNSub]] = [initial]

    for _ in range(max_steps):
        if not beam:
            break

        async def expand(node: SearchNode[HS, PSNSub]):
            expansions = await problem.expand(node)
            event_listener.received_expansions(node, expansions)
            return expansions

        candidates = [
            new_node
            for per_node_expansion in await asyncio.gather(
                *(expand(node) for node in beam)
            )
            for new_node in per_node_expansion
        ]

        # We allow `candidates` and `finished` to compete with each other,
        # as the score will no longer decrease monotonically when we have a length penalty.
        sorted_candidates_plus_finished = sorted(
            candidates + finished, key=lambda n: n.cost
        )
        beam = []
        finished = []
        for n in sorted_candidates_plus_finished[:beam_size]:
            if n.is_finished:
                finished.append(n)
            else:
                beam.append(n)

        # If there's a less-competitive candidate which is finished, then keep it for later
        if keep_finished_nodes:
            for n in sorted_candidates_plus_finished[beam_size:]:
                if n.is_finished:
                    finished_extra.append(n)

        # Due to cycles or some other reason, hidden states are not freed on
        # time unless we manually collect.
        gc.collect()

    return sorted(finished + finished_extra, key=lambda n: n.cost)[: beam_size * 2]
