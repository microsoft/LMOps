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
# Added chat templates for models (when they have examples)

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="pku-align",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep=" ",
    )
)

# UltraLM / UltraRM Chat Template
# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="openbmb",
        system_message="",
        roles=("User: ", "Assistant: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    )
)

# e.g. https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward#usage
# prefix_user = "Human:"
# prefix_bot = "\n\nAssistant:"
# query = "列举一种空气污染。"
# response = "一种常见的空气污染源是化石燃料的燃烧产生的尾气排放，包括来自汽车、卡车、飞机、
#       火车和工业厂房的废气排放。这会导致大气中的二氧化硫、氮氧化物、一氧化碳、臭氧和颗粒物（例如灰尘和烟雾）等污染物含量增加，对人类健康和环境造成不利影响。"
register_conv_template(
    Conversation(
        name="Ziya",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
    )
)
