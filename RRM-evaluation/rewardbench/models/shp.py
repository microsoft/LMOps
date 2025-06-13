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
# Instructions from readme here https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl

import random
from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class SHPPipeline:
    def __init__(self, task, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        # turn off gradients for model and set in eval mode
        self.model.eval().requires_grad_(False)

    def __call__(self, candidates_A: List[List[Dict]], candidates_B: List[List[Dict]], **kwargs):
        """
        Inputs should be the following format, where candidates_A/B are lists of lists of dicts with strings (batches):
        [
            {
                "content": "post",
                "role": "user"
            },
            {
                "content": "response",
                "role": "assisstant"
            },
            ...
        ]

        Which returns the following:

        POST: { the context, such as the 'history' column in SHP (not containing any newlines \n) }

        RESPONSE A: { first possible continuation (not containing any newlines \n) }

        RESPONSE B: { second possible continuation (not containing any newlines \n) }

        Which response is better? RESPONSE

        ---
        The way to do this is get the post as the part of response a and response b that are the same.
        Then randomize the order of response a and response b, and then tokenize the entire sequence.
        Pass it into the model, decide on winner.

        From the model readme:
        >> input_text = "POST: Instacart gave me 50 pounds of limes instead of 5 pounds...
            what the hell do I do with 50 pounds of limes? I've already donated a bunch and gave a bunch away.
            I'm planning on making a bunch of lime-themed cocktails, but... jeez. Ceviche? \n\n RESPONSE A: Lime juice,
            then freeze in small quantities.\n\n RESPONSE B: Lime marmalade lol\n\n Which response is better? RESPONSE"
        >> x = tokenizer([input_text], return_tensors='pt').input_ids.to(device)
        >> y = model.generate(x, max_new_tokens=1)
        >> tokenizer.batch_decode(y, skip_special_tokens=True)
        >> ['A'] # returns A or B

        Output will be a list of True or False if response A was chosen by the model over B
        """
        assert len(candidates_A) == len(candidates_B), "Batches of candidates A and B must have the same length"

        input_texts = []
        orders = []
        for conv_A, conv_B in zip(candidates_A, candidates_B):
            conversation = self._extract_conversation(conv_A, conv_B)
            response_A = conv_A[-1]["content"]  # Last message of A
            response_B = conv_B[-1]["content"]  # Last message of B
            formatted_input, order = self._format_input(conversation, response_A, response_B)
            input_texts.append(formatted_input)
            orders.append(order)

        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)

        tokenized_inputs = self.tokenizer(
            input_texts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model.generate(**tokenized_inputs, max_new_tokens=1)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        bools = [output == "A" for output in decoded_outputs]
        # for each order in orders, if order is BA, flip the bool in bools
        for i, order in enumerate(orders):
            if order == "BA":
                bools[i] = not bools[i]
        return torch.Tensor(bools)

    def _extract_conversation(self, conv_A: List[Dict], conv_B: List[Dict]) -> str:
        # Combine the messages in the conversation, excluding the last responses
        conversation = [msg["content"] for msg in conv_A[:-1]]  # Exclude last response
        return " ".join(conversation)

    def _format_input(self, post: str, response_A: str, response_B: str) -> str:
        # Randomize the order of responses, but keep labels (A, B) fixed
        responses = [(response_A, "A"), (response_B, "B")]
        random.shuffle(responses)

        # Keep track of the order
        order = "".join([label for _, label in responses])

        # Use fixed labels with potentially swapped response contents
        formatted_responses = f"\n\n RESPONSE A: {responses[0][0]}\n\n RESPONSE B: {responses[1][0]}"
        return f"POST: {post}{formatted_responses}\n\n Which response is better? RESPONSE", order
