# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Small helpers shared by the Ray trainer and L2C flows."""

from verl import DataProto


def compute_response_mask(data: DataProto):
    """Return the attention-mask slice corresponding to response tokens."""
    response_length = data.batch["responses"].size(1)
    return data.batch["attention_mask"][:, -response_length:]
