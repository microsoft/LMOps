# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch
from cached_property import cached_property
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.datum import Datum, DatumSub, FullDatum, FullDatumSub
from semantic_parsing_with_constrained_lm.fewshot import (
    GPT2TokenizerQuirks,
    PromptBuilder,
    TrainSelectorStage,
)
from semantic_parsing_with_constrained_lm.lm import (
    HS,
    AutoregressiveModel,
    IncrementalLanguageModel,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.search import (
    ConstrainedDecodingProblem,
    DecodingOutput,
    FullSearchNode,
    LoggingEventListener,
    PackedSearchNode,
    PartialParse,
    beam_search,
)


@dataclass
class IncrementalLMSimilarityFunction:
    model: IncrementalLanguageModel

    async def __call__(self, train_datum: FullDatum, test_datum: Datum) -> float:
        prefix_tokens = self.model.tokenizer.encode(f"1. {train_datum.natural}\n2.")
        # TODO: If test_datum.natural begins with something that will cause
        # this leading space to be its own token, should we put the space in prefix_tokens instead?
        completion_tokens = self.model.tokenizer.encode(f" {test_datum.natural}")
        return await self.model.logprob_of_completion(prefix_tokens, completion_tokens)


@dataclass(frozen=True, eq=True)
class DatumPackedSearchNode(Generic[DatumSub], PackedSearchNode):
    test_datum: DatumSub

    def append(self, token: int) -> "DatumPackedSearchNode":
        return DatumPackedSearchNode(
            tokens=self.tokens + (token,), test_datum=self.test_datum
        )


# https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class DatumProblemFactory(Generic[DatumSub, HS], ABC):
    partial_parse_builder: Callable[[DatumSub], PartialParse]
    length_normalization: float = 0.7
    top_k: Optional[int] = None
    cache: Optional[MutableMapping[PackedSearchNode, List[FullSearchNode[HS]]]] = None

    def initial(self, datum: DatumSub) -> DatumPackedSearchNode:
        return DatumPackedSearchNode(tokens=(), test_datum=datum)

    @cached_property
    def problem(self) -> ConstrainedDecodingProblem[HS, DatumPackedSearchNode]:
        return ConstrainedDecodingProblem(
            self.autoregressive_model,
            self.unpack_node,
            self.eos_tokens,
            self.length_normalization,
            self.top_k,
            self.cache,
        )

    @property
    @abstractmethod
    def autoregressive_model(self) -> AutoregressiveModel[HS]:
        pass

    @abstractmethod
    async def unpack_node(
        self, packed_node: DatumPackedSearchNode[DatumSub]
    ) -> Tuple[PartialParse, HS, DecodingOutput, Sequence[float]]:
        pass

    @property
    @abstractmethod
    def eos_tokens(self) -> torch.Tensor:
        pass


@dataclass
class FewShotLMProblemFactorySettings(Generic[FullDatumSub, DatumSub, HS]):
    train_data: List[FullDatumSub]
    train_selectors: List[TrainSelectorStage[FullDatumSub, DatumSub]]
    prompt_builder: PromptBuilder[FullDatumSub, DatumSub]

    incremental_lm: IncrementalLanguageModel[HS]
    tokenizer_quirks: GPT2TokenizerQuirks


@dataclass
class FewShotLMProblemFactory(
    DatumProblemFactory[DatumSub, HS],
    FewShotLMProblemFactorySettings[FullDatumSub, DatumSub, HS],
):
    _eos_tokens: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        eos_tokens_set: Set[int] = {
            self.incremental_lm.tokenizer.encoder[
                self.incremental_lm.tokenizer.eos_token
            ]
        }
        eos_byte_encoded = "".join(
            self.incremental_lm.tokenizer.byte_encoder[b]
            for b in self.prompt_builder.stop.encode("utf-8")
        )
        for token, i in self.incremental_lm.tokenizer.encoder.items():
            if token.startswith(eos_byte_encoded):
                eos_tokens_set.add(i)

        self._eos_tokens = torch.tensor(sorted(eos_tokens_set), dtype=torch.long)
        self.tokenizer_quirks.check_prompt_builder(self.prompt_builder)

    def initial(self, datum: DatumSub) -> DatumPackedSearchNode:
        return DatumPackedSearchNode(tokens=(), test_datum=datum)

    @property
    def autoregressive_model(self) -> AutoregressiveModel[HS]:
        return self.incremental_lm

    @property
    def eos_tokens(self) -> torch.Tensor:
        return self._eos_tokens

    async def unpack_node(
        self, packed_node: DatumPackedSearchNode[DatumSub]
    ) -> Tuple[PartialParse, HS, DecodingOutput, Sequence[float]]:
        selected_train_data: Sequence[Sequence[FullDatumSub]] = [self.train_data]
        for train_selector in self.train_selectors:
            selected_train_data = await train_selector(
                selected_train_data, packed_node.test_datum
            )
        prompt_prefixes = [
            self.prompt_builder.assemble(train_data_piece, packed_node.test_datum)
            for train_data_piece in selected_train_data
        ]
        print("Prompt prefixes:")
        for prompt_prefix in prompt_prefixes:
            print(prompt_prefix)
            print("==========")

        assert len(prompt_prefixes) == 1
        [prompt_prefix] = prompt_prefixes

        prompt_prefix_tokens = self.incremental_lm.tokenizer.encode(
            self.tokenizer_quirks.postprocess_prompt(prompt_prefix)
        )
        all_tokens = prompt_prefix_tokens + list(packed_node.tokens)
        logprobs, hidden_state = await self.incremental_lm.execute(all_tokens)

        # https://github.com/python/mypy/issues/708
        initial_partial_parse = self.partial_parse_builder(packed_node.test_datum)  # type: ignore
        for token in packed_node.tokens:
            initial_partial_parse = initial_partial_parse.append(token)

        allowed_tokens, can_end = initial_partial_parse.allowed_next(
            torch.argsort(logprobs[-1], descending=True)
        )
        # TODO: Figure out how to generalize this for when some tokens are already present
        if not packed_node.tokens:
            self.tokenizer_quirks.check_initial_allowed_tokens(
                set(allowed_tokens.tolist()) if allowed_tokens is not None else None,
                can_end,
            )
        return (
            initial_partial_parse,
            hidden_state,
            DecodingOutput(logprobs[-1], allowed_tokens, can_end,),
            logprobs[
                list(range(len(prompt_prefix_tokens), len(all_tokens))),
                list(packed_node.tokens),
            ].tolist(),
        )


@dataclass
class Seq2SeqProblemFactorySettings(Generic[DatumSub, HS]):
    seq2seq_model: Seq2SeqModel[HS]


@dataclass
class Seq2SeqProblemFactory(
    DatumProblemFactory[DatumSub, HS], Seq2SeqProblemFactorySettings[DatumSub, HS]
):
    _eos_tokens: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        self._eos_tokens = torch.tensor(
            [self.seq2seq_model.decoder_eos_id], dtype=torch.long,
        )

    @property
    def autoregressive_model(self) -> AutoregressiveModel[HS]:
        return self.seq2seq_model

    @property
    def eos_tokens(self) -> torch.Tensor:
        return self._eos_tokens

    async def unpack_node(
        self, packed_node: DatumPackedSearchNode
    ) -> Tuple[PartialParse, HS, DecodingOutput, Sequence[float]]:
        decoder_tokens = self.seq2seq_model.decoder_bos_ids + list(packed_node.tokens)
        logprobs, hidden_state = await self.seq2seq_model.initial(
            self.seq2seq_model.encode_for_encoder(packed_node.test_datum.natural),
            decoder_tokens,
        )
        # https://github.com/python/mypy/issues/708
        initial_partial_parse = self.partial_parse_builder(packed_node.test_datum)  # type: ignore
        for token in packed_node.tokens:
            initial_partial_parse = initial_partial_parse.append(token)
        allowed_tokens, can_end = initial_partial_parse.allowed_next(
            torch.argsort(logprobs[-1], descending=True)
        )

        return (
            initial_partial_parse,
            hidden_state,
            DecodingOutput(logprobs[-1], allowed_tokens, can_end),
            logprobs[
                list(
                    range(len(self.seq2seq_model.decoder_bos_ids), len(decoder_tokens))
                ),
                list(packed_node.tokens),
            ].tolist(),
        )


@dataclass
class ModelResult:
    text: str
    cost: float


class Model(Generic[DatumSub]):
    # TODO: Make the return type less specific to beam search.
    async def predict(self, test_datum: DatumSub) -> List[ModelResult]:
        raise NotImplementedError


@dataclass
class BeamSearchSemanticParser(Model[DatumSub], Generic[DatumSub, FullDatumSub, HS]):
    problem_factory: DatumProblemFactory[DatumSub, HS]
    tokenizer: GPT2Tokenizer
    # TODO: Move this into AutoregressiveModel
    finalizer: Callable[[List[int]], str]

    # Beam search-related parameters.
    # They could be moved to its own class so that we can also parametrize search methods.
    beam_size: int
    max_steps_fn: Optional[Callable[[DatumSub], Optional[int]]] = None

    async def predict(self, test_datum: DatumSub) -> List[ModelResult]:
        """Returns tuple of (hypothesis, whether hypothesis was artificially kept
        alive using force_decokde, k-best list"""
        max_steps = self.max_steps_fn(test_datum) if self.max_steps_fn else None
        results = await beam_search(
            self.problem_factory.problem,
            self.problem_factory.initial(test_datum),
            self.beam_size,
            event_listener=LoggingEventListener(self.tokenizer, self.beam_size),
            max_steps=max_steps,
        )
        return [
            ModelResult(self.finalizer(n.tokens), n.cost)  # type: ignore
            for n in results
        ]
