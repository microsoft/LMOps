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

# Script to output the per-token reward across a piece of text given a reward model

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)

from rewardbench import models

REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
    },
    "oasst": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
        "models": [
            "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
            "OpenAssistant/reward-model-deberta-v3-base",
            "OpenAssistant/reward-model-deberta-v3-large",
            "OpenAssistant/reward-model-deberta-v3-large-v2",
            "OpenAssistant/reward-model-electra-large-discriminator",
        ],
    },
    "Starling": {
        "model_builder": models.starling.build_starling_rm,
        "pipeline_builder": models.starling.StarlingPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "models": [
            "berkeley-nest/Starling-RM-7B-alpha",
        ],
    },
    "openbmb": {
        "model_builder": models.openbmb.LlamaRewardModel.from_pretrained,
        "pipeline_builder": models.openbmb.OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "models": ["openbmb/UltraRM-13b"],
    },
    "PairRM": {
        "model_builder": models.pairrm.DebertaV2Model.from_pretrained,
        "pipeline_builder": models.pairrm.PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "models": [
            "llm-blender/PairRM",
            "llm-blender/PairRM-hf",
        ],
    },
    "BetterPairRM": {
        "model_builder": models.betterpairrm.DebertaV2Model.from_pretrained,
        "pipeline_builder": models.betterpairrm.PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "models": [
            "mightbe/Better-PairRM",
        ],
    },
    "SHP": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": models.shp.SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "models": [
            "stanfordnlp/SteamSHP-flan-t5-large",
            "stanfordnlp/SteamSHP-flan-t5-xl",
        ],
    },
}


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument(
        "text",
        type=str,
        help="Text to evaluate.",
    )
    # optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="natolambert/gpt2-dummy-rm",
        help="Path to the model or HuggingFace link.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to non-matching tokenizer, requires --direct_load.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="tulu",
        help="Path to the chat template.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="per-token-reward",
        help="Directory to store the hashes and token information.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (if above number of tokens).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Input validation
    def _validate_require_pairwise_inputs(models):
        for model in models:
            if args.model in model or args.chat_template in model:
                raise ValueError(f"{model} require pairwise inputs, not supported")

    _validate_require_pairwise_inputs(models=["PairRM", "SHP"])

    return args


def main():
    args = get_args()
    model_name = args.model if args.model in REWARD_MODEL_CONFIG.keys() else "default"

    config = REWARD_MODEL_CONFIG.get(model_name)

    if args.random_seed:
        print(f"Setting random seed to {args.random_seed}")
        torch.manual_seed(args.random_seed)

    if config["custom_dialogue"]:
        raise ValueError("Custom dialogue formatting not yet supported in this script")

    # Setup the accelerate state first before using logging since it errors out
    # if you do the other first.
    accelerator = Accelerator(cpu=True)
    current_device = accelerator.process_index

    # Setup logging
    logger = setup_logging(name=__name__)
    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # Prepare dataset and tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _tokenify_string(string):
        _tokens = tokenizer.tokenize(string)
        cumulative_texts = [tokenizer.convert_tokens_to_string(_tokens[: i + 1]) for i, _ in enumerate(_tokens)]
        # Hacky approach. Ideally we can do a str.split(" ") but we want to
        # preserve the subword tokenization by the tokenizer.
        tokens = [tokenizer.convert_tokens_to_string([t]) for t in _tokens]
        return cumulative_texts, tokens

    substrings, tokens = _tokenify_string(args.text)
    dataset = Dataset.from_list([{"text": substring} for substring in substrings])

    # Load reward model pipeline
    logger.info("Loading reward model")
    reward_pipeline = load_reward_pipeline(
        args.model,
        config=config,
        tokenizer=tokenizer,
        process_index=current_device,
    )
    reward_pipeline_kwargs = {
        "batch_size": args.batch_size,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }

    # Perform inference and get per-token reward
    per_token_rewards = get_per_token_reward(
        dataset,
        reward_pipeline=reward_pipeline,
        reward_pipeline_kwargs=reward_pipeline_kwargs,
        accelerator=accelerator,
        is_custom_pipeline=config["pipeline_builder"] == pipeline,
        logger=logger,
        dataloader_batch_size=args.batch_size,
    )

    # Report the results
    for reward, span in zip(per_token_rewards, substrings):
        print(f"Reward: {round(reward, 3)} | Substring: {span}")

    # Save the results
    save_results(
        output_dir=args.output_dir,
        text=args.text,
        model=args.model,
        chat_template=args.chat_template,
        substrings=substrings,
        tokens=tokens,
        rewards=per_token_rewards,
    )


def get_config(model_name: str, default_if_missing: bool = True) -> Dict[str, Any]:
    """Get the appropriate loading configuration given a model name

    We only do minimal string matching here, basically checking if a substring, say,
    oasst or others exist in REWARD_MODEL_CONFIG.keys().

    model_name (str): the HuggingFace link or pointer to the model.
    default_if_missing (bool): if True, will return the default configuration if
        model is missing from our config templates. If False, then it raises
        a ValueError.
    RETURNS (Dict[str, Any]): the loading configuration for a given model.
    """
    for tpl, config in REWARD_MODEL_CONFIG.items():
        available_models = config["models"]
        if model_name in available_models:
            config = config.pop("models")
            print(f"Returning configuration from {tpl}. Config={config}")
            return config

    # If model_name is not found anywhere
    if default_if_missing:
        print("Model {model_name} not found in available models. Returning default configuration")
        return REWARD_MODEL_CONFIG.get("default")
    else:
        raise ValueError(f"Model {model_name} not found in available models!")


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Create a logger"""
    logger = get_logger(name or __name__)
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
    return logger


def load_reward_pipeline(
    model_name: str,
    *,
    config: Dict[str, Any],
    tokenizer: "transformers.PreTrainedTokenizer",
    process_index: int,
) -> transformers.Pipeline:
    """Load a reward model pipeline given a model configuration and its tokenizer.

    model_name (str): the HuggingFace link or pointer to the model.
    config (Dict[str, Any]): the model configuration.
    tokenizer (transformers.PreTrainedTokenizer): the tokenizer to use with the model.
    process_index (int): the machine to run the process.
    RETURNS (transformers.Pipeline) the reward model pipeline
    """
    model_kwargs = {"device_map": {"": process_index}}
    if config["quantized"]:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
            }
        )
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    if not pipeline == pipeline_builder:
        model = model_builder(model_name, **model_kwargs)
        reward_pipeline = pipeline_builder(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        reward_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=tokenizer,
            revision="main",
            model_kwargs=model_kwargs,
        )
    # Tokenization settings
    if reward_pipeline.tokenizer.pad_token_id is None:
        reward_pipeline.model.config.pad_token_id = reward_pipeline.tokenizer.eos_token_id
        reward_pipeline.tokenizer.pad_token_id = reward_pipeline.tokenizer.eos_token_id

    return reward_pipeline


def get_per_token_reward(
    dataset: Dataset,
    *,
    reward_pipeline: "transformers.Pipeline",
    reward_pipeline_kwargs: Dict[str, Any],
    accelerator: "Accelerator",
    is_custom_pipeline: bool,
    logger: "logging.Logger",
    dataloader_batch_size: int,
) -> List[float]:
    """Get the reward per subtoken

    dataset (datasets.Dataset): the HuggingFace dataset to source the text from.
    reward_pipeline (transformers.Pipeline): the reward pipeline that will provide the scores.
    accelerator (Accelerator): accelerator class for training performance.
    is_custom_pipeline (bool): flag to check if we need to run a data loader to collate the results.
    logger (logging.Logger): logger class.
    dataloader_batch_size (int): control the batch size passed to the data loader.
    RETURNS (List[float]): list of computed rewards for each token.
    """
    if is_custom_pipeline:
        logger.info("Running dataloader to collect results")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_batch_size,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        )
        dataloader, model = accelerator.prepare(dataloader, reward_pipeline.model)
        reward_pipeline.model = model

        results = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")
            rewards = reward_pipeline(batch["text"], **reward_pipeline_kwargs)
            # Some pipeline implementations return a list of dictionaries, if that's the
            # case, we only take the value in the 'score' key. Else, we just return the list.
            scores = [r["score"] for r in rewards] if isinstance(rewards[0], dict) else rewards.cpu().numpy().tolist()
            results.extend(scores)
    else:
        logger.info("Running forward pass via built-in pipeline abstraction")
        reward_pipeline = accelerator.prepare(reward_pipeline)
        results = reward_pipeline(dataset["text"], reward_pipeline_kwargs)

    return results


def save_results(
    output_dir: Path,
    text: str,
    model: str,
    chat_template: str,
    substrings: List[str],
    tokens: List[str],
    rewards: List[str],
):
    """Save results to disk

    This function will first hash the prompt, and then the model with the chat template.
    Then, it will save the model result in a JSON file on disk.

    output_dir (Path): directory to save the files.
    text (str): the text used to hash. The hashed string will be the name of the subdirectory.
    model (str): the name of the model
    chat_template (str): the name of the chat template.
    tokens (List[str]): the tokens extracted by the reward pipeline's tokenizer.
    rewards (List[str]): the rewards computed by the reward pipeline.
    """
    # Hash the text first using base16
    text_hash = hashlib.shake_256(text.encode()).hexdigest(5)
    text_dir = output_dir / text_hash
    text_dir.mkdir(parents=True, exist_ok=True)

    # Hash the model and chat_template combination
    MODEL_CHAT_DELIMITER = "___"
    model_chat_text = model + MODEL_CHAT_DELIMITER + chat_template
    model_chat_hash = hashlib.shake_256(model_chat_text.encode()).hexdigest(5)

    # Output file will be the model_chat_hash
    output_file = text_dir / f"{model_chat_hash}.json"
    print(f"Saving results to {text_dir}")

    reward_info = {
        "text": text,
        "text_hash": text_hash,
        "model": model,
        "chat_template": chat_template,
        "model_chat_hash": model_chat_hash,
        "substrings": substrings,
        "tokens": tokens,
        "rewards": rewards,
    }

    # Assumes the model output is a pointer to a HuggingFace repository
    with open(output_file, "w") as f:
        json.dump(reward_info, f, indent=4)


if __name__ == "__main__":
    main()
