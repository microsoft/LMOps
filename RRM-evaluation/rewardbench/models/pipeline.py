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

import torch


# should be redundant, but having determinism issues
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return model


class RewardBenchPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = disable_dropout_in_model(model).eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            return_tensors="pt",
        ).to("cuda")

        # if tokenizer.bos_token exists, check if there is a double bos token to start the inputs
        # if so, we'll remove the first one and pass in the inputs (somewhat hacky solution)
        # a full refactor can be done to use tokenizer.apply_chat_template(chat, tokenize=True)
        # though, so many RM implementations are non standard, so this is a quick fix rather than ecosystem wide
        if self.tokenizer.bos_token:
            bos_token_id = self.tokenizer.bos_token_id
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Ensure input_ids is 2D
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # Find the start of each sequence (first non-pad token)
            seq_starts = attention_mask.argmax(dim=1)

            # Check for double BOS tokens
            seq_second = torch.clamp(seq_starts + 1, max=input_ids.size(1) - 1)
            double_bos_mask = (input_ids[torch.arange(input_ids.size(0)), seq_starts] == bos_token_id) & (
                input_ids[torch.arange(input_ids.size(0)), seq_second] == bos_token_id
            )

            # Set attention mask to 0 for the first BOS token where double BOS is detected
            if double_bos_mask.any():
                attention_mask[
                    torch.arange(attention_mask.size(0), device=attention_mask.device)[double_bos_mask],
                    seq_starts[double_bos_mask],
                ] = torch.tensor(0, device=attention_mask.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        if return_inputs:
            return outputs.logits, inputs
        else:
            return outputs.logits
