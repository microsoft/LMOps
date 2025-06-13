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

# copied partially from https://github.com/yuchenlin/LLM-Blender/blob/main/llm_blender/pair_ranker/pairrm.py
# and added pairwise tokenization function from https://huggingface.co/llm-blender/PairRM-hf
# requires jinja2, install with pip install jinja2

from typing import List

import jinja2
from transformers import PreTrainedModel, PreTrainedTokenizer

# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


def tokenize_conv_pair(tokenizer, convAs: List[str], convBs: List[str], **kwargs):
    def truncate_texts(text, max_length, truncate_side):
        tokenizer.truncation_side = truncate_side
        tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_length)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    BETTER_PAIRRM_TEMPLATE = """{% for message in messages -%}
    {% if message['role'] == 'user' -%}
    USER: {{ message['content']|trim -}}
    {% if not loop.last -%}


    {% endif %}
    {% elif message['role'] == 'assistant' -%}
    ASSISTANT: {{ message['content']|trim -}}
    {% if not loop.last -%}


    {% endif %}
    {% elif message['role'] == 'user_context' -%}
    USER: {{ message['content']|trim -}}
    {% if not loop.last -%}


    {% endif %}
    {% elif message['role'] == 'system' -%}
    SYSTEM MESSAGE: {{ message['content']|trim -}}
    {% if not loop.last -%}


    {% endif %}
    {% endif %}
    {% endfor -%}
    {% if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
    ASSISTANT: {% endif -%}"""

    jinja2_env = jinja2.Environment()
    jinja2_template = jinja2_env.from_string(BETTER_PAIRRM_TEMPLATE)

    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all(
            [c_a[i]["content"] == c_b[i]["content"] for i in range(0, len(c_a), 2)]
        ), "USER turns must be the same"

    inputs = [
        truncate_texts(jinja2_template.render(messages=x[:-1], add_generation_prompt=True), 2030, "left")
        for x in convAs
    ]
    cand1_texts = [truncate_texts(x[-1]["content"], 670, "right") for x in convAs]
    cand2_texts = [truncate_texts(x[-1]["content"], 670, "right") for x in convBs]

    encodings = tokenize_pair(tokenizer, inputs, cand1_texts, cand2_texts, **kwargs)
    return encodings


def tokenize_pair(
    tokenizer,
    sources: List[str],
    candidate1s: List[str],
    candidate2s: List[str],
    source_prefix="<|source|>",
    cand1_prefix="<|candidate1|>",
    cand2_prefix="<|candidate2|>",
    source_max_length=2030,
    candidate_max_length=670,
    **kwargs,
):
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    source_tokens = tokenizer.encode(source_prefix)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        tokenizer.truncation_side = "left"
        source_ids = source_tokens + tokenizer.encode(
            sources[i], max_length=source_max_length - len(source_tokens), truncation=True
        )

        tokenizer.truncation_side = "right"
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(
            cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True
        )
        candidate2_ids = tokenizer.encode(
            cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True
        )
        ids.append(source_ids + candidate1_ids + candidate2_ids)

    encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
    return encodings


class BetterPairRMPipeline:
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(self, task, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # turn off gradients for model and set in eval mode
        self.model.eval().requires_grad_(False)

    def __call__(self, candidates_A: List[str], candidates_B: List[str], output_logits=False, **kwargs):
        AB_encodings = tokenize_conv_pair(self.tokenizer, candidates_A, candidates_B, **kwargs)
        AB_outputs = self.model(**AB_encodings.to(self.model.device))
        AB_logits = AB_outputs.logits
        BA_encodings = tokenize_conv_pair(self.tokenizer, candidates_B, candidates_A, **kwargs)
        BA_outputs = self.model(**BA_encodings.to(self.model.device))
        BA_logits = BA_outputs.logits
        logits = AB_logits - BA_logits
        if output_logits:
            return logits.tolist()
        else:
            return logits > 0
