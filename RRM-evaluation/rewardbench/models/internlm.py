# Copyright 2024 AllenAI. All rights reserved.
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

# Pipeline for InternLM because uses different score mechanism
# Runs like:
# score1 = model.get_score(tokenizer, chat_1)
import torch


class InternLMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model.eval()
        self.tokenizer = tokenizer

    # sourced from
    # noqa https://huggingface.co/internlm/internlm2-20b-reward/blob/6539b1df85e74019a3e36da43901e18205c572ea/modeling_internlm2.py#L1931
    def __call__(self, samples, **kwargs):
        with torch.no_grad():
            batch_input_ids = []
            attention_masks = []

            for conversation_str in samples:
                input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False)
                # add reward score token at the end of the input_ids if it is not already there
                if input_ids[0, -1] != self.model.reward_token_id:
                    input_ids = torch.cat(
                        [input_ids, torch.tensor([[self.model.reward_token_id]], dtype=torch.long)], dim=1
                    )
                input_ids = input_ids.squeeze(0)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.bool)
                batch_input_ids.append(input_ids)
                attention_masks.append(attention_mask)

            r_pad_batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            r_pad_attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=False
            )

            outputs = self.model.forward(
                input_ids=r_pad_batch_input_ids.to(self.model.device),
                attention_mask=r_pad_attention_masks.to(self.model.device),
            )
            scores = outputs[0]  # .squeeze().cpu().tolist() handled later
        return scores
