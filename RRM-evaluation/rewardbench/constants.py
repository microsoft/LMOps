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

# reference for length bias categories
LENGTH_CATEGORIES = {
    "alpacaeval-easy": "True",
    "alpacaeval-hard": "True",
    "alpacaeval-length": "Neutral",
    "donotanswer": "False",
    "hep-cpp": "Neutral",
    "hep-go": "Neutral",
    "hep-java": "Neutral",
    "hep-js": "Neutral",
    "hep-python": "Neutral",
    "hep-rust": "Neutral",
    "llmbar-adver-GPTInst": "False",
    "llmbar-adver-GPTOut": "Neutral",
    "llmbar-adver-manual": "False",
    "llmbar-adver-neighbor": "False",
    "llmbar-natural": "Neutral",
    "math-prm": "Neutral",
    "mt-bench-easy": "False",
    "mt-bench-hard": "False",
    "mt-bench-med": "Neutral",
    "refusals-dangerous": "False",
    "refusals-offensive": "False",
    "xstest-should-refuse": "False",
    "xstest-should-respond": "True",
}

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

SUBSET_NAME_TO_PAPER_READY = {
    "alpacaeval-easy": "AlpacaEval Easy",
    "alpacaeval-length": "AlpacaEval Length",
    "alpacaeval-hard": "AlpacaEval Hard",
    "mt-bench-easy": "MT Bench Easy",
    "mt-bench-med": "MT Bench Medium",
    "mt-bench-hard": "MT Bench Hard",
    "llmbar-natural": "LLMBar Natural",
    "llmbar-adver-neighbor": "LLMBar Adver. Neighbor",
    "llmbar-adver-GPTInst": "LLMBar Adver. GPTInst",
    "llmbar-adver-GPTOut": "LLMBar Adver. GPTOut",
    "llmbar-adver-manual": "LLMBar Adver. Manual",
    "refusals-dangerous": "Refusals Dangerous",
    "refusals-offensive": "Refusals Offensive",
    "xstest-should-refuse": "XSTest Should Refuse",
    "xstest-should-respond": "XSTest Should Respond",
    "donotanswer": "Do Not Answer",
    "math-prm": "PRM Math",
    "hep-cpp": "HumanEvalPack CPP",
    "hep-go": "HumanEvalPack Go",
    "hep-java": "HumanEvalPack Java",
    "hep-js": "HumanEvalPack Javascript",
    "hep-python": "HumanEvalPack Python",
    "hep-rust": "HumanEvalPack Rust",
    "anthropic_harmless": "Anthropic Harmless",
    "anthropic_helpful": "Anthropic Helpful",
    "anthropic_hhh": "Anthropic HHH",
    "mtbench_gpt4": "MT Bench GPT-4",
    "mtbench_human": "MT Bench Human",
    "shp": "SHP",
    "summarize": "Summarize",
}
