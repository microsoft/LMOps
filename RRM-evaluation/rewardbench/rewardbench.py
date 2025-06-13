# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run RewardBench (evaluate any reward model on any dataset)

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pprint import pformat
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pkg_resources
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData
from huggingface_hub.repocard import RepoCard
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_and_process_dataset,
    torch_dtype_mapping,
)


@dataclass
class Args:
    # core args
    dataset: str = "allenai/reward-bench"
    """The dataset to evaluate on."""
    split: Optional[str] = None
    """The split to evaluate on."""
    model: Optional[str] = None
    """The model to evaluate."""
    revision: Optional[str] = None
    """The model revision to evaluate."""
    ref_model: Optional[str] = None
    """The reference model to compare against."""
    tokenizer: Optional[str] = None
    """The tokenizer to use (defaults to model)."""
    chat_template: Optional[str] = None
    """The chat template to use (defaults to from tokenizer, from chattemplate)."""
    not_quantized: bool = False
    """Disable quantization for models that are quantized by default."""
    prioritize_scoring: bool = False
    """Prioritize scoring of the messages key, rather than accuracy rankings."""

    # hf saving args
    push_results_to_hub: bool = False
    """Push distribution of scores and labels to randomly generated HuggingFace dataset."""
    upload_model_metadata_to_hf: bool = False
    """Upload metadata to Hugging Face Hub."""
    hf_entity: Optional[str] = None
    """The Hugging Face entity to push results to."""
    hf_name: Optional[str] = None
    """[Default is random] The Hugging Face dataset name to push results to."""

    # wandb args
    wandb_run: Optional[str] = None
    """The wandb run to extract model and revision from."""

    # inference args
    batch_size: int = 8
    """The batch size to use."""
    max_length: int = 512
    """The max length to use."""
    torch_dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float16"
    """PyTorch dtype (default: float16)"""
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None
    """Attention implementation to use (default: None)"""

    # system args
    load_json: bool = False
    """Load dataset as json."""
    trust_remote_code: bool = False
    """Trust remote code."""
    debug: bool = True
    """Debug mode."""
    output_dir: str = "results/"
    """The output directory to save results."""
    save_all: bool = False
    """Save all results."""
    force_truncation: bool = False
    """Force truncation (for if model errors)."""


def save_jsonl(save_filename: str, table: Dict[str, List[Union[int, float, str]]]):
    # Ensure directory exists
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Write the dictionary data to JSONL file
    with open(save_filename, "w") as outfile:
        # Iterate through each index and write corresponding row as JSON
        for i in range(len(next(iter(table.values())))):  # Get the first key's length
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def push_results_to_hub(args, results, accuracy=None):
    """
    Push dataset to Hugging Face Hub.

    Args:
        args: Argument object with the following attributes:
            - hf_entity: Hugging Face entity (e.g., username or organization).
            - hf_name: ID of the repository to create or use.
    """
    api = HfApi()

    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]

    timestamp = time.strftime("%H%M%d%m%y")
    # Generate default hf_name if not set
    if not args.hf_name:
        args.hf_name = f"rewardbench_eval_{timestamp}"

    full_repo_id = f"{args.hf_entity}/{args.hf_name}"

    # Create repository on Hugging Face Hub
    api.create_repo(full_repo_id, repo_type="dataset", exist_ok=True)

    # Print and prepare the repository URL
    repo_full_url = f"https://huggingface.co/datasets/{full_repo_id}"

    # Generate the command that was run
    run_command = " ".join(["python"] + sys.argv)

    # Get package versions as a dictionary
    package_versions = {package.key: package.version for package in pkg_resources.working_set}

    # If accuracy is provided, create a string adding it to the results
    if accuracy is not None:
        accuracy_str = f"Accuracy: {accuracy}"
    else:
        accuracy_str = ""

    # Create and push a repo card
    rm_card = RepoCard(
        content=f"""\
# {args.hf_name}: RewardBench CLI Eval. Outputs

See https://github.com/allenai/reward-bench for more details

Built with the `rewardbench` CLI tool.
{accuracy_str}

Command used to run:
```
{run_command}
```

## Configs
```
args: {pformat(vars(args))}
```

## Package Versions
```
{pformat(package_versions)}
```
"""
    )
    rm_card.push_to_hub(
        full_repo_id,
        repo_type="dataset",
    )
    print(f"Pushed to {repo_full_url}")

    # Upload the dataset (after to add metadata to card)
    data_to_upload = Dataset.from_dict(results)
    data_to_upload.push_to_hub(full_repo_id)

    return full_repo_id


def main():
    print(11111, flush=True)
    parser = HfArgumentParser((Args))
    print(11111, flush=True)
    rewardbench(*parser.parse_args_into_dataclasses())


# Secondary function structure needed to accomodate HuggingFace Args with CLI binding
def rewardbench(args: Args):
    if args.wandb_run is not None:
        wandb_run = wandb.Api().run(args.wandb_run)
        args.model = wandb_run.config["hf_name"]
        args.revision = wandb_run.config["hf_repo_revision"]

    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    logger.info(f"Running on device {current_device}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # basic checks from config
    if args.ref_model:
        is_dpo = True
        MODEL_CONFIGS = DPO_MODEL_CONFIG
        assert args.model != args.ref_model, "policy and reference model should be different"
        from trl.trainer.utils import DPODataCollatorWithPadding

        from rewardbench import DPOInference
    else:
        is_dpo = False
        MODEL_CONFIGS = REWARD_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template(args.chat_template)
    else:
        conv = None
    
    print(conv)

    if args.model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model]
    else:
        config = MODEL_CONFIGS["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    if not is_dpo:
        quantized = config["quantized"]  # only Starling isn't quantized for now
        # if llama-3 in name, switch quantized to False (severely degrades performance)
        if (
            ("llama-3" in args.model)
            or ("Llama3" in args.model)
            or ("Llama-3" in args.model)
            or ("LLaMA3" in args.model)
            or args.not_quantized
        ):
            quantized = False
            logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")
        custom_dialogue = config["custom_dialogue"]
        pipeline_builder = config["pipeline_builder"]
        _ = config["model_type"]
        torch_dtype = config.get("torch_dtype", None)
        if custom_dialogue:
            raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    model_builder = config["model_builder"]

    # Handle datatype
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    #########################
    # load dataset
    #########################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code, revision=args.revision
    )
    if args.dataset == "allenai/reward-bench":
        logger.info("Running core eval dataset.")
        from rewardbench import load_eval_dataset
        from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
        from rewardbench.utils import calculate_scores_per_section

        # primary set compiles slightly more information
        dataset, subsets = load_eval_dataset(
            core_set=True,
            conv=conv,
            custom_dialogue_formatting=False,
            tokenizer=tokenizer,
            logger=logger,
            return_extra_data=True,
        )
    else:
        dataset = load_and_process_dataset(
            args.dataset,
            split=args.split,
            json=args.load_json,
            tokenizer=tokenizer,
            conv=conv,
            prioritize_instructions=args.prioritize_scoring,
        )
    print(dataset[0])
    # check if "chosen" and "rejected" in the dataset features
    if "text_chosen" in dataset.features and "text_rejected" in dataset.features:
        is_preference_ranking = True
    else:
        is_preference_ranking = False

    if args.debug:
        dataset = dataset.select(range(10))

    # Move extra columns to extra metadata (merged later)
    keep_columns = ["prompt", "text_chosen", "text_rejected"] if is_preference_ranking else ["prompt", "text"]
    all_cols = dataset.column_names
    metadata = dataset.remove_columns(keep_columns)
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    logger.info("*** Load reward model ***")

    ############################
    # Load DPO model pipeline
    ############################
    print("is_dpo", is_dpo)
    if is_dpo:
        # if not preference data, raise NotImplementedError (only implemented for pairwise)
        if not is_preference_ranking:
            raise NotImplementedError("DPO only implemented for pairwise preference data.")
        tokenizer.pad_token = tokenizer.eos_token
        # if no BOS token, set as pad token, e.g. QWEN models
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model = model_builder(
            args.model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )

        # use internal inference functions in DPO trainer
        dpo = DPOInference(
            model,
            ref_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            # norm is norm, avg is average, sum is sum
        )

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )

    ############################
    # Load classifier model pipeline
    ############################
    else:

        # padding experiments for determinism
        tokenizer.padding_side = "left"
        truncation = False
        if args.force_truncation:
            truncation = True
            tokenizer.truncation_side = "left"

        reward_pipeline_kwargs = {
            "batch_size": args.batch_size,  # eval_args.inference_batch_size,
            "truncation": truncation,
            "padding": True,
            "max_length": args.max_length,
            "function_to_apply": "none",  # Compute raw logits
            "return_token_type_ids": False,
        }
        if quantized:
            model_kwargs = {
                "load_in_8bit": True,
                "device_map": {"": current_device},
                "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
            }
        else:
            # note, device map auto does not work for bitsandbytes quantized models
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch_dtype,
            }

        # if attn_implementation is not specified, this falls back to Hugging Face's default
        # strategy (which chooses between sdpa and eager depending on pytorch version)
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        model = model_builder(
            args.model, **model_kwargs, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
        reward_pipe = pipeline_builder(
            "text-classification",  # often not used
            model=model,
            tokenizer=tokenizer,
        )

        # set pad token to eos token if not set
        if reward_pipe.tokenizer.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
            reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
        # For models whose config did not contains `pad_token_id`
        if reward_pipe.model.config.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

        # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
        if not check_tokenizer_chat_template(tokenizer):
            reward_pipe.tokenizer.add_eos_token = True

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = accelerator.prepare(reward_pipe.model)
        reward_pipe.model = model

    ############################
    # Run inference
    ############################

    results = []
    if is_preference_ranking:
        scores_chosen = []
        scores_rejected = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")
        print(is_preference_ranking, "is_preference_ranking")
        if is_preference_ranking:
            if is_dpo:
                rewards_chosen, rewards_rejected = dpo.inference_step(batch)
            else:
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)
            #print(rewards_chosen, rewards_rejected)
            
            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                score_chosen_batch = [result["score"] for result in rewards_chosen]
                score_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

            # log results
            [
                results.append(1) if chosen > rejected else results.append(0)
                for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
            ]
            scores_chosen.extend(score_chosen_batch)
            scores_rejected.extend(score_rejected_batch)
        else:
            rewards = reward_pipe(batch["text"], **reward_pipeline_kwargs)
            if isinstance(rewards[0], dict):
                scores = [result["score"] for result in rewards]
            else:
                scores = rewards.cpu().float().numpy().tolist()
            results.extend(scores)

    ############################
    # save outputs directly
    ############################

    def unwrap_if_list_of_lists(data):
        if isinstance(data, list):
            if isinstance(data[0], list):
                return [item for sublist in data for item in sublist]
        return data

    combined_data = {
        "prompt": dataset["prompt"],  # Assuming `prompts` is a list of prompts matching scores
        "results": unwrap_if_list_of_lists(results),
    }

    # Consolidate chosen and rejected scores along with prompts and texts
    if is_preference_ranking:
        combined_data["scores_chosen"] = unwrap_if_list_of_lists(scores_chosen)
        combined_data["scores_rejected"] = unwrap_if_list_of_lists(scores_rejected)
        combined_data["text_chosen"] = dataset["text_chosen"]
        combined_data["text_rejected"] = dataset["text_rejected"]
    # or take instruction
    else:
        combined_data["text"] = dataset["text"]

    # add columns in metadata to combined_data
    for col in metadata.column_names:
        combined_data[col] = metadata[col]

    # Save combined scores and metadata to JSONL
    scores_output_path = os.path.join(args.output_dir, f"{args.model}_outputs.jsonl")
    save_jsonl(scores_output_path, combined_data)

    ############################
    # the rest is just for preferences (accuracies)
    ############################
    if is_preference_ranking:
        ############################
        # compile scores
        ############################
        # calculate accuracy
        accuracy = sum(results) / len(results)
        logger.info(f"Results: {accuracy}, on {len(results)} prompts")

        # compute mean and std of scores, chosen and rejected, then margin between them
        logger.info(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
        logger.info(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
        logger.info(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

        if args.dataset == "allenai/reward-bench":
            out_dataset = dataset.add_column("results", results)
            if args.debug:
                subsets = subsets[:10]
            out_dataset = out_dataset.add_column("subsets", subsets)
            out_dataset = out_dataset.to_pandas()  # I know this is meh

            results_grouped = {}
            present_subsets = np.unique(out_dataset["subsets"])
            for subset in present_subsets:
                subset_dataset = out_dataset[out_dataset["subsets"] == subset]
                num_correct = sum(subset_dataset["results"])
                num_total = len(subset_dataset["results"])
                logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
                results_grouped[subset] = num_correct / num_total

            results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
            logger.info(f"Results: {results_section}")

        ############################
        # save scores
        ############################
        # save score in json to args.output_dir + args.model + ".json"
        output_path = args.output_dir + args.model + ".json"
        dirname = os.path.dirname(output_path)
        os.makedirs(dirname, exist_ok=True)

        # remove old data
        if os.path.exists(output_path):
            os.remove(output_path)

        final_results = {
            "accuracy": accuracy,
            "num_prompts": len(results),
            "model": args.model,
            "ref_model": args.ref_model,
            "tokenizer": tokenizer_path,
            "chat_template": args.chat_template,
            "extra_results": results_grouped if args.dataset == "allenai/reward-bench" else None,
        }
        with open(output_path, "w") as f:
            json.dump(final_results, f)

        if args.wandb_run is not None:
            for key in final_results:
                wandb_run.summary[f"rewardbench/{key}"] = final_results[key]
            wandb_run.update()
            print(f"Logged metrics to {wandb_run.url}")

        # if save_all is passed, save a large jsonl with all scores_chosen, scores_rejected
        if args.save_all:
            output_path = args.output_dir + args.model + "_all.jsonl"
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)

            # remove old data
            if os.path.exists(output_path):
                os.remove(output_path)

            with open(output_path, "w") as f:
                for chosen, rejected in zip(scores_chosen, scores_rejected):
                    f.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")

        ############################
        # Upload metadata to Hugging Face Hub
        ############################
        if args.upload_model_metadata_to_hf:
            logger.info("*** Uploading metadata to Hugging Face Hub ***")
            try:
                # Initialize ModelCardData with basic metadata
                card_data = ModelCardData(
                    language="en",
                    model_name=args.model,
                    eval_results=[
                        EvalResult(
                            task_type="preference_evaluation",
                            dataset_type=args.dataset,
                            dataset_name=args.dataset.split("/")[-1],  # Assuming dataset ID is like 'owner/dataset'
                            metric_type="accuracy",
                            metric_value=accuracy,
                        )
                    ],
                )

                # If there are extra results (per subset), add them as separate EvalResults
                if args.dataset == "allenai/reward-bench" and results_grouped:
                    for section, section_accuracy in results_section.items():
                        print(f"Adding section {section} with accuracy {section_accuracy}")
                        section_eval = EvalResult(
                            task_type="preference_evaluation",
                            dataset_type=section.replace(" ", "_"),
                            dataset_name=section,
                            metric_type="accuracy",
                            metric_value=section_accuracy,
                        )
                        card_data.eval_results.append(section_eval)

                    for subset, subset_accuracy in results_grouped.items():
                        print(f"Adding subset {subset} with accuracy {subset_accuracy}")
                        subset_eval = EvalResult(
                            task_type="preference_evaluation",
                            dataset_type=subset,
                            dataset_name=subset,
                            metric_type="accuracy",
                            metric_value=subset_accuracy,
                        )
                        card_data.eval_results.append(subset_eval)

                # Create a ModelCard
                card = ModelCard.from_template(
                    card_data,
                    model_id=args.model,
                )

                # Push the updated ModelCard to the Hugging Face Hub
                card.push_to_hub(
                    args.model, revision=args.revision, commit_message="Update evaluation results via RewardBench"
                )
                logger.info(f"Successfully pushed updated ModelCard to Hugging Face Hub for {args.model}")
            except Exception as e:
                logger.error(f"Failed to upload metadata to Hugging Face Hub: {e}")
                logger.info("(The most common issue is a model you do not have write permissions on).")
    else:
        accuracy = None

    ############################
    # Upload results to HF (as dataset)
    ############################
    if args.push_results_to_hub:
        hf_repo = push_results_to_hub(args, combined_data, accuracy=accuracy)
        logger.info(f"Pushed results to Hugging Face Hub for https://huggingface.co/datasets/{hf_repo}")


if __name__ == "__main__":
    main()
