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

# Copied partially from https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha#uses
# with modifications & fixes

import math
import os

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
)

SUPPORTED_STARLING_MODELS = ["berkeley-nest/Starling-RM-7B-alpha", "Nexusflow/Starling-RM-34B"]


def build_starling_rm(model_name, **kwargs):
    if model_name == "berkeley-nest/Starling-RM-7B-alpha":
        reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf", **kwargs)
        directory = snapshot_download(model_name)
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        # TODO, not documented by authors how to quantize this
        reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        reward_model = reward_model.to("cuda")
    elif model_name == "berkeley-nest/Starling-RM-34B":
        reward_model = LlamaForSequenceClassification.from_pretrained("berkeley-nest/Starling-RM-34B", **kwargs)
    else:
        raise ValueError(
            f"Model {model_name} not found in Starling reward models. Supported are {SUPPORTED_STARLING_MODELS}"
        )

    reward_model.eval().requires_grad_(False)
    return reward_model


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.transformer.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        scores = torch.stack(scores)
        return {"scores": scores}


class GPTRewardModel(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


class StarlingPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        """
        samples: List[str]
        """
        batch_size = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        encoding_dict = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        input_ids = encoding_dict["input_ids"]
        attention_masks = encoding_dict["attention_mask"]
        out = []
        with torch.no_grad():
            for i in range(math.ceil(len(samples) / batch_size)):
                rewards = self.model(
                    input_ids=input_ids[i * batch_size : (i + 1) * batch_size],
                    attention_mask=attention_masks[i * batch_size : (i + 1) * batch_size],
                )
                # if scores are dict (for Yi model), extract them from tensor.
                if isinstance(rewards, dict):
                    rewards = rewards["scores"]
                out.extend(rewards)

        return torch.hstack(out)
