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

__version__ = "0.1.4"
from .chattemplates import *  # noqa
from .dpo import DPOInference
from .models import DPO_MODEL_CONFIG, REWARD_MODEL_CONFIG
from .utils import (
    check_tokenizer_chat_template,
    load_and_process_dataset,
    load_bon_dataset,
    load_bon_dataset_v2,
    load_eval_dataset,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
    reroll_and_score_dataset,
    save_to_hub,
    torch_dtype_mapping,
)

__all__ = [
    check_tokenizer_chat_template,
    DPOInference,
    DPO_MODEL_CONFIG,
    load_bon_dataset,
    load_eval_dataset,
    load_and_process_dataset,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
    REWARD_MODEL_CONFIG,
    save_to_hub,
    torch_dtype_mapping,
    load_bon_dataset_v2,
    reroll_and_score_dataset,
]
