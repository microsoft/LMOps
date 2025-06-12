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

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from vllm import LLM, SamplingParams
import torch
import re
from datasets import load_from_disk, Dataset, load_dataset

from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
)
from rewardbench.utils import calculate_scores_per_section

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)
    
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument("--local_model", type=str, default=None, help="local model path")
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    parser.add_argument("--vllm_max_seq_length", type=int, default=8192, help="max sequence length for vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    parser.add_argument("--model_modifier", type=str, default=None, help="model modifier for special cases")
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=2048,
        help="max number of response tokens to generate (for vllm)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for sampling (for vllm)")
    parser.add_argument("--dataset", type=str, default=None, help="dataset to use (for vllm)")
    parser.add_argument("--run_name", type=str, default=None, help="run name for output files")
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    args = parser.parse_args()
    return args

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def second_last_boxed_only_string(string):
    # 找最后一个 \boxed 或 \fbox
    last_idx = string.rfind("\\boxed")
    if last_idx < 0:
        last_idx = string.rfind("\\fbox")
    
    if last_idx < 0:  # 仍然没找到，返回 None
        return None

    # 找倒数第二个
    second_last_idx = string.rfind("\\boxed", 0, last_idx)
    if second_last_idx < 0:
        second_last_idx = string.rfind("\\fbox", 0, last_idx)
    
    if second_last_idx < 0:  # 仍然没找到，返回 None
        return None

    # 解析倒数第二个 \boxed 或 \fbox 里的内容
    i = second_last_idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    else:
        return string[second_last_idx:right_brace_idx + 1]


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def extract_boxed_answer_baseline(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = second_last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def extract_answer_baseline(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer_baseline(passage)
    return None

def extract_answer_judgelm(passage: str) -> list:
    # 提取第一个换行符前的文本片段
    first_line = passage.split('\n')[0]
    # 在该段落中找出最多两个数字
    numbers = re.findall(r'\d+', first_line)[:2]
    return [int(n) for n in numbers]

def clean_value(val):
    if isinstance(val, bool):
        return str(val)  # 或者 return "" 代表空字符串
    return val

def main():
    args = get_args()
    np.random.seed(args.seed)
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)


    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST
        
    if args.local_model is not None:
        checkpoint_path = os.path.join(args.local_model, "data.pt")
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model = AutoModelForCausalLM.from_pretrained(args.model)
    # if model isn't API, load via vllm
    if not is_api_models:
        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # load model
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            seed=args.seed,
            #max_seq_length=args.vllm_max_seq_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.model_modifier == "JudgeLM":
            tokenizer.model_max_length = 8192
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None
            
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            print("✅ 该 tokenizer 带有 chat_template")
        #else:
            #tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}"
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=1,
            max_tokens=args.max_response_tokens,
            stop_token_ids=stop_token_ids,
        )

    # handle off-case models
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
    elif "Con-J" in args.model:
        model_modifier = "Con-J"
    elif "OffsetBias" in args.model:
        model_modifier = "offsetbias"
    elif "Atla" in args.model:
        logger.info("Using ATLA model")
        model_modifier = "Atla"
    elif "gemini" in args.model:
        model_modifier = "gemini"
    elif "RISE-Judge" in args.model:
        model_modifier = "RISE-Judge"
    else:
        model_modifier = None
    model_modifier = args.model_modifier
    print(f"Model modifier: {model_modifier}")
    #model_modifier = "Skywork"
    #model_modifier = "baseline"
    ############################
    # Load dataset
    ############################
    if args.dataset == "rewardbench":
        logger.info("*** Load dataset ***")
        dataset, subsets = load_eval_dataset(
            core_set=not args.pref_sets,
            conv=get_conv_template("raw"),  # not used in this script (handled later)
            custom_dialogue_formatting=True,  # handle formatting later
            tokenizer=None,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id"],
            max_turns=4,
        )
        #print(type(dataset))

        # copy id for saving, then remove
        ids = dataset["id"]
        dataset = dataset.remove_columns("id")

        # debug: use only 10 examples
        if args.debug:
            dataset = dataset.select(range(10))
            subsets = subsets[:10]
            ids = ids[:10]
    
    elif args.dataset == "MATH":
        if args.run_name.startswith("MATH_7b_round_1") or args.run_name.startswith("MATH_32b_round_1"):
            group_num = 0
        elif args.run_name.startswith("MATH_7b_round_2") or args.run_name.startswith("MATH_32b_round_2"):
            group_num = 1
        elif args.run_name.startswith("MATH_7b_round_3") or args.run_name.startswith("MATH_32b_round_3"):
            group_num = 2
        elif args.run_name.startswith("MATH_7b_round_4") or args.run_name.startswith("MATH_32b_round_4"):
            group_num = 3
        elif args.run_name.startswith("MATH_7b_round_5") or args.run_name.startswith("MATH_32b_round_5"):
            group_num = 4
        dataset = load_dataset("lmarena-ai/PPE-MATH-Best-of-K", split="train")
        scores = dataset["scores"]
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        #print(columns.keys())

        dataset = Dataset.from_dict(columns)
        formatted_dataset = []
        
        with open(f"./MATH_relative_data/extracted_conflict_pairs.json", "r") as f:
            previous_round_results = json.load(f)
        
        for idx, example in enumerate(dataset):
            rival_1 = previous_round_results[idx][group_num][0]
            rival_2 = previous_round_results[idx][group_num][1]
            rival_1_score = scores[idx][rival_1]
            prompt = example["prompt"]
            _, after_question = prompt.split('<problem>', 1)
            question_with_options = after_question.strip()
            if rival_1_score:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
            else:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
            formatted_dataset.append({
                "text_chosen": [chosen_example, assistant_chosen_example],
                "text_rejected": [rejected_example, assistant_rejected_example]
            })
            #print(f"idx: {idx}, rival_1: {rival_1}, rival_2: {rival_2}")
        dataset = formatted_dataset
        #print(dataset[0])
            
        subsets = ["MATH"] * len(dataset)
        ids = list(range(len(dataset)))

        # debug: use only 10 examples
        if args.debug:
            dataset = dataset[:20]
            subsets = subsets[:20]
            ids = ids[:20]
    
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        dataset = Dataset.from_dict(columns)
    elif args.dataset == "MMLU":
        if args.run_name.startswith("MMLU_7b_round_1") or args.run_name.startswith("MMLU_32b_round_1"):
            group_num = 0
        elif args.run_name.startswith("MMLU_7b_round_2") or args.run_name.startswith("MMLU_32b_round_2"):
            group_num = 1
        elif args.run_name.startswith("MMLU_7b_round_3") or args.run_name.startswith("MMLU_32b_round_3"):
            group_num = 2
        elif args.run_name.startswith("MMLU_7b_round_4") or args.run_name.startswith("MMLU_32b_round_4"):
            group_num = 3
        elif args.run_name.startswith("MMLU_7b_round_5") or args.run_name.startswith("MMLU_32b_round_5"):
            group_num = 4
        dataset = load_dataset("lmarena-ai/PPE-MMLU-Pro-Best-of-K", split="train")
        scores = dataset["scores"]
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        #print(columns.keys())

        dataset = Dataset.from_dict(columns)
        formatted_dataset = []
        
        with open(f"./MMLU_relative_data/extracted_conflict_pairs.json", "r") as f:
            previous_round_results = json.load(f)
        
        for idx, example in enumerate(dataset):
            rival_1 = previous_round_results[idx][group_num][0]
            rival_2 = previous_round_results[idx][group_num][1]
            rival_1_score = scores[idx][rival_1]
            prompt = example["prompt"]
            _, after_question = prompt.split('Question:', 1)
            question_with_options = after_question.strip()
            if rival_1_score:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
            else:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
            formatted_dataset.append({
                "text_chosen": [chosen_example, assistant_chosen_example],
                "text_rejected": [rejected_example, assistant_rejected_example]
            })
            #print(f"idx: {idx}, rival_1: {rival_1}, rival_2: {rival_2}")
        dataset = formatted_dataset
        #print(dataset[0])
            
        subsets = ["MMLU"] * len(dataset)
        ids = list(range(len(dataset)))

        # debug: use only 10 examples
        if args.debug:
            dataset = dataset[:20]
            subsets = subsets[:20]
            ids = ids[:20]
    
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        dataset = Dataset.from_dict(columns)
    elif args.dataset == "GPQA":
        if args.run_name.startswith("GPQA_7b_round_1") or args.run_name.startswith("GPQA_32b_round_1"):
            group_num = 0
        elif args.run_name.startswith("GPQA_7b_round_2") or args.run_name.startswith("GPQA_32b_round_2"):
            group_num = 1
        elif args.run_name.startswith("GPQA_7b_round_3") or args.run_name.startswith("GPQA_32b_round_3"):
            group_num = 2
        elif args.run_name.startswith("GPQA_7b_round_4") or args.run_name.startswith("GPQA_32b_round_4"):
            group_num = 3
        elif args.run_name.startswith("GPQA_7b_round_5") or args.run_name.startswith("GPQA_32b_round_5"):
            group_num = 4
        dataset = load_dataset("lmarena-ai/PPE-GPQA-Best-of-K", split="train")
        scores = dataset["scores"]
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        print(columns.keys())

        dataset = Dataset.from_dict(columns)
        formatted_dataset = []
        
        with open(f"./GPQA_relative_data/extracted_conflict_pairs.json", "r") as f:
            previous_round_results = json.load(f)
        
        for idx, example in enumerate(dataset):
            rival_1 = previous_round_results[idx][group_num][0]
            rival_2 = previous_round_results[idx][group_num][1]
            rival_1_score = scores[idx][rival_1]
            prompt = example["prompt"]
            _, after_question = prompt.split('Question', 1)
            question_with_options = after_question.strip()
            if rival_1_score:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
            else:
                chosen_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_chosen_example = {
                    "content": example[f"response_{rival_2+1}"],
                    "role": "assistant"
                }
                
                rejected_example = {
                    "content": question_with_options,
                    "role": "user"
                }
                assistant_rejected_example = {
                    "content": example[f"response_{rival_1+1}"],
                    "role": "assistant"
                }
            formatted_dataset.append({
                "text_chosen": [chosen_example, assistant_chosen_example],
                "text_rejected": [rejected_example, assistant_rejected_example]
            })
        dataset = formatted_dataset
            
        subsets = ["GPQA"] * len(dataset)
        ids = list(range(len(dataset)))

        # debug: use only 10 examples
        if args.debug:
            dataset = dataset[:20]
            subsets = subsets[:20]
            ids = ids[:20]
    
        columns = {key: [example[key] for example in dataset] for key in dataset[0]}
        dataset = Dataset.from_dict(columns)
    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement(batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                winner, request, judgement = run_judge_pair(
                    prompt, answer_a, answer_b, args.model, multi_turn=mult_turn, model_modifier=model_modifier
                )
                if debug:
                    print(f"Prompt: {request}")
                    print(f"Judgement: {judgement}")

                # handle voting
                if isinstance(winner, list):
                    # print votes if debug
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:  # if "error"
                    return 0.5  # effectively a tie
            else:
                return 0.5

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm
        ############################

        def format_judgements(batch, optional_chat_template=None):
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            if args.dataset == "rewardbench":
                local_seed = 100 * args.seed + hash(prompt) % 10000
            else:
                local_seed = 10 * rival_1 + rival_2 + hash(prompt) % 10000 + args.seed * 1000
            np.random.seed(local_seed)
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
            system_prompt, user_prompt = format_judge_answers(
                prompt, answer_a, answer_b, multi_turn=mult_turn, model_modifier=model_modifier
            )    
            if optional_chat_template is not None:
                optional_chat_template.set_system_message(system_prompt)
                optional_chat_template.messages = []
                optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
                optional_chat_template.append_message(optional_chat_template.roles[1], None)
                prompt = optional_chat_template.get_prompt()
            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = system_prompt + user_prompt
                # chat template already include special tokens
                # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                # so we need to tokenize without adding special tokens
                tokenized_prompt = tokenizer(prompt, add_special_tokens=False, return_length=True)
                prompt_ids = tokenized_prompt["input_ids"]
            #print(system_prompt)
            #print(user_prompt)
            batch["text"] = prompt
            batch["is_shuffled"] = is_shuffled
            batch["prompt_ids"] = prompt_ids
            return batch

        # format the dataset for the model, with optional fastchat templating
        if args.chat_template is not None:
            chat_template = get_conv_template(args.chat_template)
        else:
            chat_template = None
        dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})
        # collect texts of dataset in list
        prompts = dataset_prompts["text"]
        
        prompt_ids = dataset_prompts["prompt_ids"]
        is_shuffled = dataset_prompts["is_shuffled"]

        # generate
        logger.info("*** Run inference ***")
        if model_modifier == "Atla":
            logger.info("Using Atla model for inference")
            outputs = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
        else:
            outputs = model.generate(prompts, sampling_params=sampling_params)
        logger.info("*** Inference done ***")
        answers = [o.outputs[0].text for o in outputs]
        model_answers = []
        if model_modifier == "Skywork":
            for answer in answers:
                answer = THOUGHT_DELIMITER_START + answer
                if THOUGHT_DELIMITER_START in answer and THOUGHT_DELIMITER_END in answer:
                    model_solution = answer.split(THOUGHT_DELIMITER_END)[1]
                    model_answer = extract_answer(model_solution)
                    model_answers.append(model_answer)
                else:
                    model_answers.append(3)
         
        winners = [process_judgement(a, model_modifier) for a in model_answers]

        def process_shuffled(win, shuffle):
            if shuffle:
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:  # if "error"
                return 0.5  # effectively a tie

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

        min_len = min(len(prompts), len(answers), len(model_answers), len(winners), len(results))

        data = [
            {
                "prompt": prompts[i],
                "answer": answers[i],
                "model_answer": model_answers[i],
                "winner": winners[i],
                "result": results[i],
                "is_shuffled": is_shuffled[i],
            }
            for i in range(min_len)
        ]

        # 存入 JSON 文件
        if args.run_name.startswith("MATH_7b"):
            with open(f"./results/7b_MATH_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("MATH_32b"):
            with open(f"./results/32b_MATH_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("MMLU_7b"):
            with open(f"./results/7b_MMLU_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("MMLU_32b"):    
            with open(f"./results/32b_MMLU_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("GPQA_7b"):
            with open(f"./results/7b_GPQA_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("GPQA_32b"):
            with open(f"./results/32b_GPQA_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("RewardBench_7b"):
            with open(f"./results/7b_RewardBench_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        if args.run_name.startswith("RewardBench_32b"):
            with open(f"./results/32b_RewardBench_fivepair/{args.run_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        

        print(f"Saved {min_len} entries to output.json")

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # model name concat if list
    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    result_list = []
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total
        result_list.append({"subset": subset, "score": num_correct / num_total})

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)
        results_grouped["leaderboard"] = results_leaderboard
    
    # save to json
    if args.run_name.startswith("MATH_7b"):
        with open(f"./results/7b_MATH_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("MATH_32b"):
        with open(f"./results/32b_MATH_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("MMLU_7b"):
        with open(f"./results/7b_MMLU_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("MMLU_32b"):
        with open(f"./results/32b_MMLU_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("GPQA_7b"):
        with open(f"./results/7b_GPQA_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("GPQA_32b"):
        with open(f"./results/32b_GPQA_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("RewardBench_7b"):
        with open(f"./results/7b_RewardBench_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)
    if args.run_name.startswith("RewardBench_32b"):
        with open(f"./results/32b_RewardBench_fivepair/{args.run_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(results_grouped, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
