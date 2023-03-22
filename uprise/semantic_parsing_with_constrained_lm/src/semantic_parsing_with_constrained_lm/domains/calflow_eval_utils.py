# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Functions to help run Bart cold monster model for Calflow. """
import asyncio
from typing import List, Optional

import torch
import tqdm

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.configs.lib.calflow import make_semantic_parser_for_calflow
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow import CalflowOutputLanguage
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import ClientType
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_bart import Seq2SeqBart
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.model import BeamSearchSemanticParser, ModelResult


def instantiate_bart_eval_model(
    model_loc: str, grammar_dir: Optional[str] = None,
) -> BeamSearchSemanticParser:
    preprocessed_grammar = PreprocessedGrammar.from_folder(grammar_dir)
    lm = Seq2SeqBart(
        model_loc,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    beam_size = 2
    return make_semantic_parser_for_calflow(
        [],
        lm,
        use_gpt3=False,
        beam_size=beam_size,
        output_type=CalflowOutputLanguage.Canonical,
        client_type=ClientType.BART,
        preprocessed_grammar=preprocessed_grammar,
        constrained=True,
        num_examples_per_prompt=0,
    )


def predict(model: BeamSearchSemanticParser, user_utterance: str) -> List[str]:
    results: List[ModelResult] = asyncio.run(
        model.predict(
            Datum(
                natural=user_utterance,
                dialogue_id=None,
                turn_part_index=None,
                agent_context=None,
            )
        )
    )
    if len(results) == 0:
        return []
    return [result.text for result in results]


def evaluate(eval_examples: List[FullDatum], model: BeamSearchSemanticParser) -> None:
    total = len(eval_examples)
    correct = 0
    for example in tqdm.tqdm(eval_examples):
        predicted = predict(model, example.natural)[0]
        if (
            example.canonical is not None
            and example.canonical.strip() == predicted.strip()
        ):
            correct += 1
        else:
            print(f"Utterance: {example.natural}")
            print(f"Canonical: {example.canonical}")
            print(f"Predicted: {predicted}")

    acc = correct * 1.0 / total
    for _ in range(100):
        print(f"Accuracy = {correct} / {total} = {acc}")
