# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, cast

import torch
from cached_property import cached_property
from transformers import BartForConditionalGeneration, BartTokenizer, GPT2Tokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.async_tools.batch_helper import BatchingHelper, BatchMaker
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import Seq2SeqHelper, Seq2SeqModel


@dataclass
class BartState:
    encoder_tokens: Tuple[int, ...]
    decoder_tokens: Tuple[int, ...]
    # Shape: (sequence_length, embed_size)
    encoder_outputs: torch.Tensor = dataclasses.field(repr=False)
    # Tuple of tuple(torch.FloatTensor) of length config.n_layers,
    # with each tuple having 2 tensors of shape
    #   (num_heads, sequence_length, embed_size_per_head)
    # and 2 additional tensors of shape
    #   (num_heads, encoder_sequence_length, embed_size_per_head).
    past_key_values: Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...
    ] = dataclasses.field(repr=False)


@dataclass(eq=True, frozen=True)
class BartBatchMaker(BatchMaker):
    input_length: int
    output_length: int
    uses_hidden_state: bool

    model: BartForConditionalGeneration = dataclasses.field(compare=False)

    @property
    def max_batch_size(self) -> int:
        # TODO: Revisit this entirely arbitrary number
        return 10

    @property
    def timeout(self) -> float:
        return 0.001

    @classmethod
    def from_args(
        cls,
        model: BartForConditionalGeneration,
        args: Tuple[Sequence[int], Sequence[int], Optional[BartState]],
    ):
        encoder_tokens, decoder_tokens, hidden_state = args
        if hidden_state is None:
            return cls(
                len(encoder_tokens),
                len(decoder_tokens),
                uses_hidden_state=False,
                model=model,
            )
        else:
            assert len(encoder_tokens) == 0
            return cls(
                len(hidden_state.encoder_tokens),
                len(hidden_state.decoder_tokens) + len(decoder_tokens),
                uses_hidden_state=True,
                model=model,
            )

    async def execute(
        self, args: List[Tuple[Sequence[int], Sequence[int], Optional[BartState]]]
    ) -> Tuple[torch.Tensor, List[BartState]]:
        if self.uses_hidden_state:
            return await self._execute_with_hidden_state(args)
        else:
            return await self._execute_without_hidden_state(args)

    async def _execute_without_hidden_state(
        self, args: List[Tuple[Sequence[int], Sequence[int], Optional[BartState]]]
    ) -> Tuple[torch.Tensor, List[BartState]]:
        encoder_tokens, decoder_tokens, _ = zip(*args)
        input_ids = torch.tensor(
            list(encoder_tokens), dtype=torch.long, device=self.model.device
        )
        decoder_input_ids = torch.tensor(
            list(decoder_tokens), dtype=torch.long, device=self.model.device
        )

        model_outputs = self.model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        )
        logprobs = torch.nn.functional.log_softmax(model_outputs["logits"], dim=-1)

        next_hidden_states = [
            BartState(
                tuple(encoder_tokens[i]),
                tuple(decoder_tokens[i]),
                model_outputs["encoder_last_hidden_state"][i],
                tuple(
                    cast(
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                        tuple(
                            past_key_values_internal[i]
                            for past_key_values_internal in past_key_values_for_layer
                        ),
                    )
                    for past_key_values_for_layer in model_outputs["past_key_values"]
                ),
            )
            for i in range(len(args))
        ]
        return logprobs, next_hidden_states

    async def _execute_with_hidden_state(
        self, args: List[Tuple[Sequence[int], Sequence[int], Optional[BartState]]]
    ):
        _, decoder_tokens, hidden_states = zip(*args)
        decoder_input_ids = torch.tensor(
            list(decoder_tokens), dtype=torch.long, device=self.model.device
        )
        encoder_outputs = torch.stack(
            [hidden_state.encoder_outputs for hidden_state in hidden_states]
        )
        past_key_values = tuple(
            tuple(
                torch.stack(
                    [
                        hidden_state.past_key_values[layer_i][kv_i]
                        for hidden_state in hidden_states
                    ]
                )
                for kv_i in range(4)
            )
            for layer_i in range(len(hidden_states[0].past_key_values))
        )

        model_outputs = self.model(
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_outputs,),
            past_key_values=past_key_values,
        )
        logprobs = torch.nn.functional.log_softmax(model_outputs["logits"], dim=-1)

        next_hidden_states = [
            BartState(
                past_hidden_state.encoder_tokens,
                past_hidden_state.decoder_tokens + decoder_tokens,
                model_outputs["encoder_last_hidden_state"][i],
                tuple(
                    cast(
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                        tuple(
                            past_key_values_internal[i]
                            for past_key_values_internal in past_key_values_for_layer
                        ),
                    )
                    for past_key_values_for_layer in model_outputs["past_key_values"]
                ),
            )
            for i, past_hidden_state in enumerate(hidden_states)
        ]
        return logprobs, next_hidden_states


@dataclass
class Seq2SeqBart(Seq2SeqModel[BartState]):
    pretrained_model_dir: str
    device: torch.device

    batch_helper: BatchingHelper[
        Tuple[Sequence[int], Sequence[int], Optional[BartState]],
        Tuple[torch.Tensor, List[BartState]],
    ] = dataclasses.field(init=False)
    model: BartForConditionalGeneration = dataclasses.field(init=False)
    seq2seq_helper: Seq2SeqHelper = dataclasses.field(init=False)

    def __post_init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(
            self.pretrained_model_dir
        )
        self.model.to(self.device)
        self.batch_helper = BatchingHelper(
            lambda args: BartBatchMaker.from_args(self.model, args),
        )

        with open(
            os.path.join(self.pretrained_model_dir, "seq2seq_settings.json")
        ) as settings_f:
            self.seq2seq_helper = Seq2SeqHelper.from_settings_json(
                settings_f.read(), self.tokenizer
            )

    @cached_property
    def tokenizer(self) -> GPT2Tokenizer:  # pylint: disable=invalid-overridden-method
        return BartTokenizer.from_pretrained(self.pretrained_model_dir)

    @cached_property
    def vocab_size(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.tokenizer.vocab_size

    @property
    def decoder_bos_ids(self) -> List[int]:
        return self.seq2seq_helper.decoder_start_token_ids

    @property
    def decoder_eos_id(self) -> int:
        return self.seq2seq_helper.decoder_eos_token_id

    def encode_for_encoder(self, s: str) -> List[int]:
        return self.seq2seq_helper.encode_for_encoder(s)

    def encode_prefix_for_decoder(
        self, s: str, include_bos_ids: bool = True
    ) -> List[int]:
        return self.seq2seq_helper.encode_prefix_for_decoder(s, include_bos_ids)

    def decode_output(self, ids: Sequence[int]) -> str:
        return self.seq2seq_helper.decode_output(ids)

    async def initial(
        self,
        encoder_tokens: Sequence[int],
        decoder_tokens: Sequence[int],
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BartState]]:
        (batched_logprobs, next_hidden_states), i = await self.batch_helper.execute(
            (encoder_tokens, decoder_tokens, None)
        )
        return (
            batched_logprobs[i, : len(decoder_tokens)],
            None if drop_next_hidden_state else next_hidden_states[i],
        )

    async def extend(
        self,
        tokens: Sequence[int],
        hidden_state: BartState,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BartState]]:
        (batched_logprobs, next_hidden_states), i = await self.batch_helper.execute(
            ([], tokens, hidden_state)
        )
        return (
            batched_logprobs[i, : len(tokens)],
            None if drop_next_hidden_state else next_hidden_states[i],
        )
