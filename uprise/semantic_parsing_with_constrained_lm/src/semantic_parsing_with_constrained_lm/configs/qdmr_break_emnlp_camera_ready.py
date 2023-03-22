# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This config file is for running experiments needed for the EMNLP camera ready.

It will generate the following experiments (depending on the value of eval_split and model):
- 100 dev examples
	- GPT-3 Constrained Canonical, n = 1000
	- GPT-3 Constrained Canonical, n = 100
	- GPT-3 Constrained Canonical, n = 25
	- GPT-3 Constrained Canonical, n = 200
	- GPT-3 Constrained Meaning, n = 200
	- GPT-3 Unconstrained Canonical, n = 200
	- GPT-3 Unconstrained Meaning, n = 200
- All dev examples
	- GPT-3 Constrained Meaning, n = 200
	- BART Constrained Canonical
	- BART Constrained Meaning
	- BART Unconstrained Canonical
	- BART Unconstrained Meaning
	- GPT-2 Constrained Canonical
	- GPT-2 Constrained Meaning
	- GPT-2 Unconstrained Canonical
	- GPT-2 Unconstrained Meaning
"""

from typing import Any, Callable, Dict

import torch
from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.configs.lib.common import PromptOrder, make_semantic_parser
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import Datum
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.qdmr_break import (
    BreakDataType,
    BreakDatum,
    BreakMetrics,
    BreakPieces,
    BreakSamplingType,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.fit_max_steps import compute_and_print_fit
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import TRAINED_MODEL_DIR, AutoregressiveModel, ClientType
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_bart import Seq2SeqBart
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.run_exp import EvalSplit, Experiment
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.search import PartialParse, StartsWithSpacePartialParse


def build_config(
    log_dir,  # pylint: disable=unused-argument
    eval_split: EvalSplit,
    model: ClientType,
    rank: int,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> Dict[str, Callable[[], Experiment]]:
    BEAM_SIZE = 10
    DEV_SUBSET_SIZE = 100
    MAX_STEPS_FOR_COMPLETION = 145

    use_gpt3 = model == ClientType.GPT3

    def create_exp(
        problem_type: Literal[
            "constrained", "unconstrained-beam", "unconstrained-greedy"
        ],
        output_type: BreakDataType,
        train_size: int,
        exp_name: str,
    ):
        lm: AutoregressiveModel
        if model == ClientType.GPT3:
            lm = IncrementalOpenAIGPT3()
        elif model == ClientType.BART:
            lm = Seq2SeqBart(
                # Part after / is set to match lm_finetune.py
                f"{TRAINED_MODEL_DIR}/20000/break_{output_type}/",
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )
        else:
            raise ValueError(model)

        piece = BreakPieces.build(
            tokenizer=lm.tokenizer,
            data_type=output_type,
            train_sampling_type=BreakSamplingType.proportional,
            test_sampling_type=BreakSamplingType.random,
            train_total=train_size,
            test_total=DEV_SUBSET_SIZE,
            seed=0,
        )
        train_data = piece.train_data
        test_data = piece.test_data

        if eval_split == EvalSplit.TrainSubset:
            piece = BreakPieces.build(
                tokenizer=lm.tokenizer,
                data_type=output_type,
                train_sampling_type=BreakSamplingType.proportional,
                test_sampling_type=BreakSamplingType.random,
                train_total=100000,
                test_total=1,
                seed=0,
            )
            test_data = piece.train_data[-100:]
        elif eval_split == EvalSplit.DevFull:
            piece = BreakPieces.build(
                tokenizer=lm.tokenizer,
                data_type=output_type,
                train_sampling_type=BreakSamplingType.proportional,
                test_sampling_type=BreakSamplingType.random,
                train_total=train_size,
                test_total=1000000,
                seed=0,
                skip_if_needed=False,
            )
            test_data = piece.test_data
        elif eval_split == EvalSplit.DevSubset:
            # train_data and test_data were already set outside of this if block
            pass
        else:
            raise ValueError(f"{eval_split} not supported currently")

        partial_parse_builder: Callable[[BreakDatum], PartialParse]
        if problem_type == "constrained":
            partial_parse_builder = piece.partial_parse_builder  # type: ignore
            beam_size = BEAM_SIZE
        elif problem_type.startswith("unconstrained"):
            # TODO: Only impose this if we are using a GPT-2-style tokenizer
            partial_parse = StartsWithSpacePartialParse(lm.tokenizer)
            partial_parse_builder = lambda _: partial_parse
            if problem_type == "unconstrained-beam":
                beam_size = BEAM_SIZE
            elif problem_type == "unconstrained-greedy":
                beam_size = 1
            else:
                raise ValueError(problem_type)
        else:
            raise ValueError(f"{problem_type} not allowed")

        # Compute max_steps_fn
        pairs = []
        for d in train_data:
            num_input_tokens = len(lm.tokenizer.tokenize(d.natural))
            num_output_tokens = len(lm.tokenizer.tokenize(d.canonical)) + 1
            pairs.append((num_input_tokens, num_output_tokens))
        max_steps_intercept, max_steps_slope = compute_and_print_fit(pairs, 10, 3)

        def max_steps_fn(datum: Datum) -> int:
            return min(
                int(
                    len(lm.tokenizer.tokenize(datum.natural)) * max_steps_slope
                    + max_steps_intercept
                ),
                MAX_STEPS_FOR_COMPLETION,
            )

        parser = make_semantic_parser(
            train_data,
            lm,
            use_gpt3,
            MAX_STEPS_FOR_COMPLETION,
            beam_size,
            partial_parse_builder,
            max_steps_fn,
            PromptOrder.BestLast,
        )

        return Experiment(  # type: ignore
            model=parser,
            metrics={
                "break_metrics": BreakMetrics(
                    log_dir=log_dir / exp_name / str(rank),
                    data_type=piece.data_type,
                    num_results=BEAM_SIZE,
                ),
            },
            test_data=test_data,
            client=lm,
        )

    def add_exp_to_dict(
        exps_dict: Dict[str, Callable[[], Experiment]],
        problem_type: Literal[
            "constrained", "unconstrained-beam", "unconstrained-greedy"
        ],
        output_type: BreakDataType,
        train_size: int,
    ):
        exp_name = (
            f"break_{model}_{eval_split}_{problem_type}_{output_type}_train{train_size}"
        )
        exps_dict[exp_name] = lambda: create_exp(
            problem_type, output_type, train_size, exp_name
        )

    result: Dict[str, Callable[[], Experiment]] = {}
    if eval_split == EvalSplit.DevFull:
        if use_gpt3:
            # - GPT-3 Constrained Meaning, n = 200
            add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=200)
        else:
            # - BART Constrained Canonical
            # - BART Constrained Meaning
            # - BART Unconstrained Canonical
            # - BART Unconstrained Meaning
            # - GPT-2 Constrained Canonical
            # - GPT-2 Constrained Meaning
            # - GPT-2 Unconstrained Canonical
            # - GPT-2 Unconstrained Meaning
            add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=200)
            add_exp_to_dict(result, "constrained", BreakDataType.qdmr, train_size=200)
            add_exp_to_dict(
                result, "unconstrained-greedy", BreakDataType.nested, train_size=200
            )
            add_exp_to_dict(
                result, "unconstrained-greedy", BreakDataType.qdmr, train_size=200
            )
    elif eval_split == EvalSplit.DevSubset:
        if use_gpt3:
            # - GPT-3 Constrained Canonical, n = 1000
            # - GPT-3 Constrained Canonical, n = 100
            # - GPT-3 Constrained Canonical, n = 25
            add_exp_to_dict(
                result, "constrained", BreakDataType.nested, train_size=1000
            )
            add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=100)
            add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=25)

            # - GPT-3 Constrained Canonical, n = 200
            # - GPT-3 Constrained Meaning, n = 200
            # - GPT-3 Unconstrained Canonical, n = 200
            # - GPT-3 Unconstrained Meaning, n = 200
            add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=200)
            add_exp_to_dict(result, "constrained", BreakDataType.qdmr, train_size=200)
            add_exp_to_dict(
                result, "unconstrained-greedy", BreakDataType.nested, train_size=200
            )
            add_exp_to_dict(
                result, "unconstrained-greedy", BreakDataType.qdmr, train_size=200
            )
        else:
            # No subset experiments for BART and GPT-2
            pass
    elif eval_split == EvalSplit.TrainSubset:
        add_exp_to_dict(result, "constrained", BreakDataType.nested, train_size=200)
        add_exp_to_dict(result, "constrained", BreakDataType.qdmr, train_size=200)
        add_exp_to_dict(
            result, "unconstrained-greedy", BreakDataType.nested, train_size=200
        )
        add_exp_to_dict(
            result, "unconstrained-greedy", BreakDataType.qdmr, train_size=200
        )

    return result
