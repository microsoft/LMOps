# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This config file is for running experiments for the EMNLP camera ready.

It will generate the following experiments (depending on the value of eval_split and model):
- 200 dev examples
	- GPT-3 Constrained Canonical, P = 20
	- GPT-3 Constrained Meaning, P = 20
	- GPT-3 Unconstrained Canonical, P = 20
	- GPT-3 Unconstrained Meaning, P = 20
	- GPT-3 Constrained Canonical, P = 8
	- GPT-3 Constrained Meaning, P = 8
- All dev examples
	- GPT-3 Constrained Canonical, P = 20
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

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.configs.lib.calflow import (
    cached_read_calflow_jsonl,
    make_semantic_parser_for_calflow,
)
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.configs.lib.common import PromptOrder
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow import CalflowMetrics, CalflowOutputLanguage
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.eval import TopKExactMatch
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm import TRAINED_MODEL_DIR, AutoregressiveModel, ClientType
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_bart import Seq2SeqBart
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.paths import DOMAINS_DIR
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.run_exp import EvalSplit, Experiment


def build_config(
    log_dir,  # pylint: disable=unused-argument
    eval_split: EvalSplit,
    model: ClientType,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> Dict[str, Callable[[], Experiment]]:

    EXAMPLES_DIR = DOMAINS_DIR / "calflow/data"
    TRAIN_SIZE = 300
    BEAM_SIZE = 10
    use_gpt3 = model == ClientType.GPT3
    preprocessed_grammar = PreprocessedGrammar.from_folder(
        str(DOMAINS_DIR / "calflow/grammar")
    )
    scfg = SCFG(preprocessed_grammar)

    def create_exp(
        problem_type: Literal[
            "constrained", "unconstrained-beam", "unconstrained-greedy"
        ],
        output_type: CalflowOutputLanguage,
    ):
        train_data = cached_read_calflow_jsonl(
            EXAMPLES_DIR / "train_300_stratified.jsonl", output_type,
        )[:TRAIN_SIZE]

        if eval_split == EvalSplit.DevFull:
            test_data = cached_read_calflow_jsonl(
                EXAMPLES_DIR / "dev_all.jsonl", output_type,
            )
        elif eval_split == EvalSplit.DevSubset:
            test_data = cached_read_calflow_jsonl(
                EXAMPLES_DIR / "test_200_uniform.jsonl", output_type,
            )
        elif eval_split == EvalSplit.TrainSubset:
            # Select a subset not already present in train
            ids_train_300 = set()
            with open(EXAMPLES_DIR / "ids_train_300_stratified.txt", "r") as id_file:
                for _, line in enumerate(id_file):
                    dialogue_id, turn_index = line.strip().split(",")
                    ids_train_300.add((dialogue_id.strip(), int(turn_index.strip())))
            train_data_1000_stratified = cached_read_calflow_jsonl(
                EXAMPLES_DIR / "train_1000_stratified.jsonl", output_type,
            )
            test_data = [
                datum
                for datum in train_data_1000_stratified
                if (datum.dialogue_id, datum.turn_part_index) not in ids_train_300
            ]
            test_data = test_data[:100]
        else:
            raise ValueError(eval_split)

        lm: AutoregressiveModel
        if model == ClientType.GPT3:
            lm = IncrementalOpenAIGPT3()
        elif model == ClientType.BART:
            lm = Seq2SeqBart(
                # Part after / is set to match lm_finetune.py
                f"{TRAINED_MODEL_DIR}/20000/calflow_{output_type}/",
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )
        else:
            raise ValueError(model)

        if problem_type == "constrained":
            constrained = True
            beam_size = BEAM_SIZE
        elif problem_type == "unconstrained-beam":
            constrained = False
            beam_size = BEAM_SIZE
        elif problem_type == "unconstrained-greedy":
            constrained = False
            beam_size = 1
        else:
            raise ValueError(f"{problem_type} not allowed")

        parser = make_semantic_parser_for_calflow(
            train_data,
            lm,
            use_gpt3,
            beam_size,
            output_type,
            model,
            preprocessed_grammar,
            constrained,
            prompt_order=PromptOrder.Shuffle,
        )

        return Experiment(
            model=parser,
            metrics={
                "exact_match": TopKExactMatch(beam_size),
                "round_trip": CalflowMetrics(
                    k=beam_size,
                    scfg=scfg,
                    data_type=output_type,
                    require_exact_length=True,
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
        output_type: CalflowOutputLanguage,
        num_examples_per_prompt: int,
    ):
        exp_name = f"calflow_{model}_{eval_split}_{problem_type}_{output_type}_prompt{num_examples_per_prompt}"
        exps_dict[exp_name] = lambda: create_exp(problem_type, output_type)

    result: Dict[str, Callable[[], Experiment]] = {}
    if eval_split == EvalSplit.DevFull:
        if use_gpt3:
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Canonical, 20)
        else:
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Canonical, 0)
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Lispress, 0)
            add_exp_to_dict(
                result, "unconstrained-greedy", CalflowOutputLanguage.Canonical, 0
            )
            add_exp_to_dict(
                result, "unconstrained-greedy", CalflowOutputLanguage.Lispress, 0
            )
    elif eval_split == EvalSplit.DevSubset:
        if use_gpt3:
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Canonical, 20)
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Lispress, 20)
            add_exp_to_dict(
                result, "unconstrained-greedy", CalflowOutputLanguage.Canonical, 20
            )
            add_exp_to_dict(
                result, "unconstrained-greedy", CalflowOutputLanguage.Lispress, 20
            )
        else:
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Canonical, 0)
            add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Lispress, 0)
    elif eval_split == EvalSplit.TrainSubset and not use_gpt3:
        add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Canonical, 0)
        add_exp_to_dict(result, "constrained", CalflowOutputLanguage.Lispress, 0)
    return result
