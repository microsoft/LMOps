# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Finetunes BART and GPT2 model with the canonical utterance generation task. """
import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple, Union, cast

import jsons
import torch
import typer
from torch.utils.data.dataset import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
)

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow import (
    CalflowOutputLanguage,
    read_calflow_jsonl,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.overnight import OutputType, OvernightPieces
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.qdmr_break import (
    BreakDataType,
    BreakPieces,
    BreakSamplingType,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.fewshot import PromptBuilder
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import TRAINED_MODEL_DIR, Seq2SeqSettings, Surround
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.paths import DOMAINS_DIR
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.finetune.batch_sampler import BucketBatchSampler

TokensAndMasksWithoutDecoder = Tuple[List[str], List[int], List[int]]
TokensAndMasksWithDecoder = Tuple[List[str], List[int], List[int], List[str]]
TokensAndMasks = Union[TokensAndMasksWithoutDecoder, TokensAndMasksWithDecoder]

PRETRAINED_MODEL_DIR = os.environ["PRETRAINED_MODEL_DIR"]


def token_level_loss(
    input_token_ids: torch.Tensor, lm_logits: torch.Tensor
) -> torch.Tensor:
    """Same as above but returns the loss tensor"""
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = input_token_ids[..., 1:].contiguous()
    # Flatten the tokens
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    return loss


class ModelForFineTuning(str, Enum):
    GPT2 = "GPT2"
    # Encoder tokens: "Human", ":", " natural", " utterance", "\n"
    # Decoder inputs: "Computer", ":", " canonical", " utterance"
    # Decoder labels: ":", " canonical", " utterance", "\n"
    BART = "Bart"
    # Encoder tokens: "<s>", "natural", " utterance", "</s>"
    # Decoder inputs: "</s>", "<s>", "canonical", " utterance"
    # Decoder labels: "<s>", <canonical", " utterance", "</s>"
    BARTv2 = "BartV2"
    # Encoder tokens: "<s>", " natural", " utterance", "</s>"
    # Decoder inputs: "</s>", "<s>", " canonical", " utterance"
    # Decoder labels: "<s>", " canonical", " utterance", "</s>"
    BARTv3 = "BartV3"


@dataclass
class ModelPieces:
    model: PreTrainedModel
    tokenizer: GPT2Tokenizer
    model_type: ModelForFineTuning
    new_line_bpe_char: str

    @staticmethod
    def from_model_type(model_type: ModelForFineTuning) -> "ModelPieces":
        if model_type.startswith("Bart"):
            model = BartForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_DIR)
            model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            tokenizer = BartTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
        elif model_type == ModelForFineTuning.GPT2:
            # Only uses GPUs if more than 4 available
            model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_DIR)
            if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
                device_map = {
                    0: list(range(12)),
                    1: list(range(12, 24)),
                    2: list(range(24, 36)),
                    3: list(range(36, 48)),
                }
                model.parallelize(device_map)  # Splits the model across several devices
            else:
                model.to(torch.device("cpu"))
            tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
        else:
            raise ValueError(model_type)
        new_line_bpe_char = tokenizer.byte_encoder[ord("\n")]
        return ModelPieces(model, tokenizer, model_type, new_line_bpe_char)


class NoamOpt:
    """Optim wrapper that implements Noam learning rate schedule."""

    def __init__(
        self,
        model_size: int,
        factor: float,
        warmup: int,
        optimizer: torch.optim.Optimizer,
    ):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.0

    def step(self) -> None:
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (
            self.warmup ** 0.5 * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class MultiplicativeOpt:
    """Optim wrapper that implements linear warmup followed by multiplicative decay."""

    def __init__(
        self,
        max_lr: float,
        warmup: int,
        steps_per_decay: int,
        optimizer: torch.optim.Optimizer,
        lr_min: float = 1e-9,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup = warmup
        self.steps_per_decay = steps_per_decay
        self._step = 0
        self._rate = 0.0
        self.lr_min = lr_min

    def step(self) -> None:
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """Implement `lrate` above"""
        if step is None:
            step = self._step

        if step < self.warmup:
            rate = self.max_lr * step / self.warmup
        elif step % self.steps_per_decay == 0:
            rate = self._rate * 0.999
        else:
            rate = self._rate

        return max(self.lr_min, rate)


def get_std_opt(
    model: PreTrainedModel, max_lr: float, warmup_steps: int, steps_per_decay: int
) -> MultiplicativeOpt:
    return MultiplicativeOpt(
        max_lr=max_lr,
        warmup=warmup_steps,
        steps_per_decay=steps_per_decay,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        ),
    )


class SimpleDataset(Dataset):
    def __init__(self, data: List[FullDatum]):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


def get_input_tokens(
    datum: FullDatum, mp: ModelPieces, use_context: bool = False
) -> TokensAndMasks:
    prompt_builder = PromptBuilder.for_demo(
        do_include_context=use_context, use_preamble=False,
    )
    utt_with_context = prompt_builder.assemble(
        [],
        Datum(
            dialogue_id=None,
            turn_part_index=None,
            natural=datum.natural,
            agent_context=datum.agent_context,
        ),
    ).rstrip(" ")
    canonical_utt = " " + datum.canonical
    if mp.model_type == ModelForFineTuning.BART:
        prompt = utt_with_context + canonical_utt
        split_index = prompt.index("\n")
        if split_index == -1:
            print(
                "newline not found, can't split prompt into encoder and decoder inputs"
            )
        encoder_input = prompt[:split_index]
        decoder_input = prompt[split_index + 1 :]
        encoder_tokens = mp.tokenizer.tokenize(encoder_input) + [mp.new_line_bpe_char]
        decoder_tokens = mp.tokenizer.tokenize(decoder_input) + [mp.new_line_bpe_char]
        attention_mask = [1] * len(encoder_tokens)
        loss_attention_mask = [1] * len(decoder_tokens)
        return (
            encoder_tokens,
            attention_mask,
            loss_attention_mask,
            decoder_tokens,
        )
    elif mp.model_type in (ModelForFineTuning.BARTv2, ModelForFineTuning.BARTv3):
        if mp.model_type == ModelForFineTuning.BARTv2:
            natural = datum.natural
            canonical = datum.canonical
        else:
            natural = " " + datum.natural
            canonical = " " + datum.canonical
        assert not use_context
        encoder_tokens = (
            [mp.tokenizer.bos_token]
            + mp.tokenizer.tokenize(natural)
            + [mp.tokenizer.eos_token]
        )
        decoder_tokens = (
            [
                mp.tokenizer.convert_ids_to_tokens(
                    mp.model.config.decoder_start_token_id
                ),
                mp.tokenizer.bos_token,
            ]
            + mp.tokenizer.tokenize(canonical)
            + [mp.tokenizer.eos_token]
        )
        attention_mask = [1] * len(encoder_tokens)
        loss_attention_mask = [1] * len(decoder_tokens)
        return (
            encoder_tokens,
            attention_mask,
            loss_attention_mask,
            decoder_tokens,
        )
    elif mp.model_type == ModelForFineTuning.GPT2:
        canonical_utt_len = len(mp.tokenizer.tokenize(canonical_utt))
        all_input_tokens = mp.tokenizer.tokenize(utt_with_context + canonical_utt) + [
            mp.new_line_bpe_char
        ]
        loss_attention_mask = [0] * (len(all_input_tokens) - canonical_utt_len - 3) + [
            1
        ] * (canonical_utt_len + 3)
        attention_mask = [1] * len(all_input_tokens)
        return all_input_tokens, attention_mask, loss_attention_mask
    else:
        raise ValueError(mp.model_type)


def get_single_batch(
    data: List[FullDatum],
    mp: ModelPieces,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_context=False,
) -> List[torch.Tensor]:
    """
    Converts FullDatum list to data and mask tensors for a single batch.
    Returns tuple containing input ids tensor, attention mask tensor,
    and a mask tensor to select tokens for the loss. If separate_encoder_decoder=True, also
    returns decoder input ids tensor, and decoder attention mask tensor.
    """
    batch_token_ids: List[
        List[int]
    ] = []  # Each list element is token ids for a FullDatum
    batch_attention_mask: List[
        List[int]
    ] = []  # Each list element is attention mask for a FullDatum
    # Each list element of loss_attention_mask is a mask over tokens we want to consider for the loss.
    # In addition to pad tokens, we also mask out tokens in the input prompt.
    batch_loss_attention_mask: List[List[int]] = []
    batch_decoder_token_ids: List[List[int]] = []

    for datum in data:
        inputs = get_input_tokens(datum, mp, use_context=use_context)
        if mp.model_type.startswith("Bart"):
            (
                datum_tokens,
                datum_attention_mask,
                datum_loss_attention_mask,
                datum_decoder_tokens,
            ) = cast(TokensAndMasksWithDecoder, inputs)
            batch_attention_mask.append(datum_attention_mask)
            batch_loss_attention_mask.append(datum_loss_attention_mask)
            batch_token_ids.append(mp.tokenizer.convert_tokens_to_ids(datum_tokens))
            batch_decoder_token_ids.append(
                mp.tokenizer.convert_tokens_to_ids(datum_decoder_tokens)
            )
        else:
            (  # pylint: disable=W0632
                datum_tokens,
                datum_attention_mask,
                datum_loss_attention_mask,
            ) = cast(TokensAndMasksWithoutDecoder, inputs)
            batch_attention_mask.append(datum_attention_mask)
            batch_loss_attention_mask.append(datum_loss_attention_mask)
            batch_token_ids.append(mp.tokenizer.convert_tokens_to_ids(datum_tokens))

    max_length = max([len(token_ids) for token_ids in batch_token_ids])
    max_decoder_length = (
        max([len(token_ids) for token_ids in batch_decoder_token_ids])
        if len(batch_decoder_token_ids) > 0
        else 0
    )
    for index, _ in enumerate(batch_token_ids):
        length = len(batch_token_ids[index])
        pad_length = max_length - length
        batch_token_ids[index] += [mp.tokenizer.pad_token_id] * pad_length
        batch_attention_mask[index] += [0] * pad_length
        if mp.model_type.startswith("Bart"):
            decoder_length = len(batch_decoder_token_ids[index])
            decoder_pad_length = max_decoder_length - decoder_length
            batch_loss_attention_mask[index] += [0] * decoder_pad_length
            batch_decoder_token_ids[index] += [
                mp.tokenizer.pad_token_id
            ] * decoder_pad_length
        else:
            batch_loss_attention_mask[index] += [0] * pad_length

    if mp.model_type.startswith("Bart"):
        outputs = [
            torch.tensor(batch_token_ids),
            torch.tensor(batch_attention_mask),
            torch.tensor(batch_loss_attention_mask),
            torch.tensor(batch_decoder_token_ids),
        ]
    else:
        outputs = [
            torch.tensor(batch_token_ids),
            torch.tensor(batch_attention_mask),
            torch.tensor(batch_loss_attention_mask),
        ]

    return [t.to(device) for t in outputs]


def get_loss(
    model,
    input_token_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    loss_attention_mask: Optional[torch.Tensor],
    decoder_token_ids: Optional[torch.Tensor] = None,
):
    """Computes the loss function using the LM probability"""
    if decoder_token_ids is None:
        outputs = model(input_token_ids, attention_mask=attention_mask)
        batch_size = input_token_ids.shape[0]
        lm_logits = outputs[0]
        token_level_loss_val = token_level_loss(input_token_ids, lm_logits)
    else:
        outputs = model(
            input_token_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_token_ids,
        )
        batch_size = input_token_ids.shape[0]
        lm_logits = outputs[0]
        token_level_loss_val = token_level_loss(decoder_token_ids, lm_logits)

    return (
        torch.sum(token_level_loss_val * loss_attention_mask[..., :-1].reshape(-1))
        / batch_size
    )


def train_model(
    mp: ModelPieces,
    exp_name: str,
    train_data: Iterable[Any],
    eval_data: Iterable[Any],
    num_steps: int,
    lr: float,
    warmup_steps: int,
    steps_per_decay: int,
    steps_per_eval: int,
    steps_per_display: int,
    steps_per_save: int,
    batch_size: int,
    use_context: bool,
    save_loc: Optional[str] = None,
):
    """
    Main train function.
    train_data and eval_data can be List[FullDatum] or List[BreakDatum].
    rank: id of GPU cluster running a single training pass. (each cluster can include multiple GPUs.)
    world_size: total number of parallel training passes being run by distributed training.
    """
    train_batch_sampler = BucketBatchSampler(
        batch_size=batch_size, bucket_width=50, shuffle_buffer_size=1000,
    )
    eval_batch_sampler = BucketBatchSampler(
        batch_size=batch_size, bucket_width=50, shuffle_buffer_size=1000,
    )
    # This needs to be consistent with what's done in prompt_builder and get_input_tokens.
    seq2seq_settings: Optional[Seq2SeqSettings] = None
    if mp.model_type in (ModelForFineTuning.GPT2, ModelForFineTuning.BART):
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(bos="Human: ", eos="\n"),
            output_surround=Surround(bos="Computer: ", eos="\n"),
        )
    elif mp.model_type == ModelForFineTuning.BARTv2:
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(
                bos=[mp.tokenizer.bos_token_id], eos=[mp.tokenizer.eos_token_id]
            ),
            output_surround=Surround(
                bos=[mp.model.config.decoder_start_token_id, mp.tokenizer.bos_token_id],
                eos=[mp.tokenizer.eos_token_id],
            ),
        )
    elif mp.model_type == ModelForFineTuning.BARTv3:
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(bos="<s> ", eos=[mp.tokenizer.eos_token_id]),
            output_surround=Surround(bos="</s><s> ", eos=[mp.tokenizer.eos_token_id],),
        )

    # Model training
    optimizer = get_std_opt(mp.model, lr, warmup_steps, steps_per_decay)
    print(f"{exp_name} {lr} {steps_per_decay}: steps {num_steps}.")
    step = 0
    mp.model.train()
    for train_datum in train_batch_sampler.batch(itertools.cycle(train_data)):
        train_inputs = get_single_batch(train_datum, mp, use_context=use_context)
        if mp.model_type.startswith("Bart"):
            (
                train_input_ids,
                train_attention_mask,
                train_loss_attention_mask,
                train_decoder_input_ids,
            ) = train_inputs
            train_loss = get_loss(
                mp.model,
                train_input_ids,
                attention_mask=train_attention_mask,
                loss_attention_mask=train_loss_attention_mask,
                decoder_token_ids=train_decoder_input_ids,
            )
        else:
            (
                train_input_ids,
                train_attention_mask,
                train_loss_attention_mask,
            ) = train_inputs
            train_loss = get_loss(
                mp.model,
                train_input_ids,
                attention_mask=train_attention_mask,
                loss_attention_mask=train_loss_attention_mask,
            )

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(mp.model.parameters(), 10)
        optimizer.step()
        step += 1

        if step % steps_per_display == 0:
            print(
                f"{datetime.now()} {batch_size} {lr} {steps_per_decay}: Step {step} || LR {optimizer.rate()} "
                f"|| Train loss for batch {train_loss.item()}"
            )

        if step % steps_per_eval == 0:
            print(
                f"{exp_name} {lr} {steps_per_decay}: {len(train_batch_sampler.batch_sizes_generated)} "
                f"batches with average size {train_batch_sampler.get_average_batch_size()}"
            )
            print(f"{exp_name} {lr} {steps_per_decay}: Evaluating ...")
            mp.model.eval()
            eval_loss = 0.0
            for eval_datum in eval_batch_sampler.batch(eval_data):
                eval_inputs = get_single_batch(eval_datum, mp, use_context=use_context)
                if mp.model_type.startswith("Bart"):
                    (
                        eval_input_ids,
                        eval_attention_mask,
                        eval_loss_attention_mask,
                        eval_decoder_input_ids,
                    ) = eval_inputs
                    eval_loss += get_loss(
                        mp.model,
                        eval_input_ids,
                        attention_mask=eval_attention_mask,
                        loss_attention_mask=eval_loss_attention_mask,
                        decoder_token_ids=eval_decoder_input_ids,
                    ).item()
                else:
                    (
                        eval_input_ids,
                        eval_attention_mask,
                        eval_loss_attention_mask,
                    ) = eval_inputs
                    eval_loss += get_loss(
                        mp.model,
                        eval_input_ids,
                        attention_mask=eval_attention_mask,
                        loss_attention_mask=eval_loss_attention_mask,
                    ).item()

            print(
                f"{exp_name} {lr}: Step {step} || LR {optimizer.rate()} || Eval loss {eval_loss}"
            )
            mp.model.train()
            print(f"{exp_name} {lr}: Done")

        if (step % steps_per_save == 0 and step > 0) or step == num_steps:
            if save_loc is None:
                model_dir = f"{TRAINED_MODEL_DIR}/{step}/{exp_name}"
            else:
                model_dir = f"{save_loc}/{step}/{exp_name}"
            print(f"{exp_name} {lr}: Saving trained model to {model_dir}")
            mp.model.eval()
            if not os.path.exists(f"{model_dir}"):
                os.makedirs(f"{model_dir}", exist_ok=True)
            mp.model.save_pretrained(model_dir)
            mp.tokenizer.save_pretrained(model_dir)
            if seq2seq_settings is not None:
                with open(f"{model_dir}/seq2seq_settings.json", "w") as settings_f:
                    settings_f.write(jsons.dumps(seq2seq_settings))

            print(f"{exp_name} {lr}: Done")

        if step >= num_steps:
            break

    if num_steps == 0:
        if save_loc is None:
            model_dir = f"{TRAINED_MODEL_DIR}/0/{exp_name}"
        else:
            model_dir = f"{save_loc}/0/{exp_name}"
        print(f"{exp_name} {lr}: Saving trained model to {model_dir}")
        mp.model.eval()
        if not os.path.exists(f"{model_dir}"):
            os.makedirs(f"{model_dir}", exist_ok=True)
        mp.model.save_pretrained(model_dir)
        mp.tokenizer.save_pretrained(model_dir)
        if seq2seq_settings is not None:
            with open(f"{model_dir}/seq2seq_settings.json", "w") as settings_f:
                settings_f.write(jsons.dumps(seq2seq_settings))
        print(f"{exp_name} {lr}: Done")

    del optimizer
    torch.cuda.empty_cache()


def main(
    lr: float = typer.Option(1e-5),
    exp_names: Optional[List[str]] = typer.Option(None),
    num_steps: int = typer.Option(20000),
    warmup_steps: int = typer.Option(1000),
    steps_per_decay: int = typer.Option(2),
    steps_per_eval: int = typer.Option(1000),
    steps_per_display: int = typer.Option(100),
    steps_per_save: int = typer.Option(20000),
    batch_size: int = typer.Option(32),
    calflow_train_data_file: Path = typer.Option(
        DOMAINS_DIR / "calflow/data/train_300_stratified.jsonl",
    ),
    calflow_eval_data_file: Path = typer.Option(
        DOMAINS_DIR / "calflow/data/dev_100_uniform.jsonl",
    ),
    use_context: bool = typer.Option(False),
    model_type: ModelForFineTuning = typer.Option(ModelForFineTuning.BARTv3),
    calflow_filter_by_ids_file: Optional[str] = typer.Option(None),
):
    model_pieces = ModelPieces.from_model_type(model_type)
    train_eval_output_list: List[Tuple[str, List[Any], List[Any]]] = []
    for calflow_data_type in list(CalflowOutputLanguage):
        exp_name = f"calflow_{calflow_data_type}"
        if exp_name not in exp_names:
            continue

        print(f"{exp_name} {lr}: Loading data for exp name {exp_name}")
        train_data = read_calflow_jsonl(calflow_train_data_file, calflow_data_type,)
        print(f"Train data read {len(train_data)}")
        if calflow_filter_by_ids_file is not None:
            ids: Set[Tuple[str, int]] = set()
            with open(calflow_filter_by_ids_file, "r") as id_file:
                for _, line in enumerate(id_file):
                    dialogue_id, turn_index = line.strip().split(",")
                    ids.add((dialogue_id.strip(), int(turn_index.strip())))
            train_data = [
                datum
                for datum in train_data
                if (datum.dialogue_id, datum.turn_part_index) in ids
            ]
        print(f"Train data after filtering {len(train_data)}")
        eval_data = read_calflow_jsonl(calflow_eval_data_file, calflow_data_type,)[:100]
        train_eval_output_list.append((exp_name, train_data, eval_data))

    # Overnight experiments
    domains = (
        "housing",
        "calendar",
        "socialnetwork",
        "basketball",
        "blocks",
        "publications",
        "recipes",
        "restaurants",
    )
    for domain in domains:
        for overnight_data_type in [
            OutputType.Utterance,
            OutputType.MeaningRepresentation,
        ]:
            exp_name = f"overnight_{domain}_{overnight_data_type}"
            if exp_name not in exp_names:
                continue

            print(f"{exp_name} {lr}: Loading data for exp name {exp_name}")
            overnight_pieces = OvernightPieces.from_dir(
                model_pieces.tokenizer,
                DOMAINS_DIR / "overnight/data",
                domain,
                is_dev=True,
                k=1,
                output_type=overnight_data_type,
                simplify_logical_forms=True,
                prefix_with_space=True,
            )
            train_eval_output_list.append(
                (
                    exp_name,
                    overnight_pieces.train_data[:200],
                    overnight_pieces.test_data[:100],
                )
            )

    # Break experiments
    for break_data_type in list(BreakDataType):
        exp_name = f"break_{break_data_type.value}"
        if exp_name not in exp_names:
            continue

        print(f"{exp_name} {lr}: Loading data for exp name {exp_name}")
        break_pieces = BreakPieces.build(
            model_pieces.tokenizer,
            data_type=break_data_type,
            train_sampling_type=BreakSamplingType.proportional,
            test_sampling_type=BreakSamplingType.random,
            train_total=200,
            test_total=100,
            seed=0,
        )
        train_eval_output_list.append(
            (exp_name, break_pieces.train_data, break_pieces.test_data,)
        )

    for exp_name, train_data, eval_data in train_eval_output_list:
        print(
            f"{exp_name} {lr}: Training on {len(train_data)} examples; saving to {TRAINED_MODEL_DIR}; "
            f"for {num_steps} steps lr {lr}"
        )
        train_data = [datum for datum in train_data if datum.canonical is not None]
        eval_data = [datum for datum in eval_data if datum.canonical is not None]
        train_model(
            model_pieces,
            exp_name,
            train_data,
            eval_data,
            num_steps,
            lr,
            warmup_steps,
            steps_per_decay,
            steps_per_eval,
            steps_per_display,
            steps_per_save,
            batch_size,
            use_context,
        )


if __name__ == "__main__":
    typer.run(main)
