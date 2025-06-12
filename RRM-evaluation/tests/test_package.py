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

# tests to make sure the code in the package is working as expected
import unittest

from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer

from rewardbench import load_and_process_dataset


class LoadAnyDataTest(unittest.TestCase):
    """
    Simple scripts to make sure the loading scripts do not error.
    """

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/rlhf-test-tokenizer")
        self.conv = get_conv_template("tulu")

    def test_load_standard_tokenizer(self):
        load_and_process_dataset(
            "allenai/ultrafeedback_binarized_cleaned", split="test_prefs", tokenizer=self.tokenizer
        )

    def test_load_standard_conv(self):
        load_and_process_dataset("allenai/ultrafeedback_binarized_cleaned", split="test_prefs", conv=self.conv)

    def test_load_alt_tokenizer(self):
        load_and_process_dataset("allenai/preference-test-sets", split="shp", tokenizer=self.tokenizer)

    def test_load_alt_conv(self):
        load_and_process_dataset("allenai/preference-test-sets", split="shp", conv=self.conv)

    def test_load_sft_tokenizer(self):
        load_and_process_dataset("HuggingFaceH4/no_robots", split="test", tokenizer=self.tokenizer)

    def test_load_sft_conv(self):
        load_and_process_dataset("HuggingFaceH4/no_robots", split="test", conv=self.conv)
