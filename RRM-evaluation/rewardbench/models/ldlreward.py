import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download
from transformers import Gemma2Model, Gemma2PreTrainedModel
from transformers.utils import ModelOutput


class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[4096, 4096]):
        super(MultiOutputNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU())

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.network(x)
        return self.softmax(x.view(x.size(0), -1, 10))


class GatingNN(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=4096, num_layers=2, temperature=1.0, dropout_prob=0.0, softmax=False
    ):
        super(GatingNN, self).__init__()
        self.temperature = temperature
        self.softmax = softmax
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_prob))

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        if self.softmax:
            x = F.softmax(x / self.temperature, dim=1)
        return x


@dataclass
class CustomOutput(ModelOutput):
    rewards: torch.FloatTensor = None
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    score: Optional[torch.FloatTensor] = None
    total_reward_distribution: Optional[torch.FloatTensor] = None
    weights: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LDLRewardModel27B(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 220)
        self.regression_layer = MultiOutputNN(config.hidden_size, self.num_objectives)
        self.gating_layer = GatingNN(
            config.hidden_size,
            self.num_objectives // 10,
            temperature=config_dict.get("temperature", 1.0),
            softmax=config_dict.get("softmax", False),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tokens_hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(tokens_hidden_states.device)
            else:
                sequence_lengths = -1

        dummy_iterator = torch.arange(batch_size, device=tokens_hidden_states.device)
        hidden_states = tokens_hidden_states[dummy_iterator, sequence_lengths]
        assert hidden_states.shape == (batch_size, self.config.hidden_size)
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            rewards = self.regression_layer(hidden_states)
            weights = self.gating_layer(hidden_states)
            weights = weights.unsqueeze(1)
            total_reward_distribution = torch.bmm(weights, rewards).squeeze(1)
            score = (
                total_reward_distribution
                * torch.linspace(0, 1, total_reward_distribution.size(-1)).to(tokens_hidden_states.device)
            ).sum(-1)
        return CustomOutput(
            rewards=rewards,
            weights=weights,
            hidden_state=hidden_states,
            total_reward_distribution=total_reward_distribution,
            score=score,
            logits=score,
        )

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory, dtype=torch.bfloat16)
        torch.save(self.regression_layer.state_dict(), os.path.join(save_directory, "regression_layer.pt"))
        torch.save(self.gating_layer.state_dict(), os.path.join(save_directory, "gating_layer.pt"))
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, device_map=None, *model_args, **kwargs):
        if not os.path.exists(load_directory):
            cached_dir = snapshot_download(repo_id=load_directory)
        else:
            cached_dir = load_directory
        model = super(LDLRewardModel27B, cls).from_pretrained(
            cached_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )

        model.regression_layer = model.regression_layer.float()
        regression_layer_path = os.path.join(cached_dir, "regression_layer.pt")
        regression_layer_state_dict = torch.load(regression_layer_path, map_location="cpu")
        model.regression_layer.load_state_dict(regression_layer_state_dict)

        model.gating_layer = model.gating_layer.float()
        gating_layer_path = os.path.join(cached_dir, "gating_layer.pt")
        gating_layer_state_dict = torch.load(gating_layer_path, map_location="cpu")
        model.gating_layer.load_state_dict(gating_layer_state_dict)

        if device_map == "auto" or device_map == "balanced":
            max_memory = get_balanced_memory(model, no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"])
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"],
                max_memory=max_memory,
            )
            model = dispatch_model(model, device_map=device_map)
        elif device_map is not None:
            raise NotImplementedError("Write your own device map")

        return model


class LDLPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model.eval()
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

            if double_bos_mask.any():
                inputs["attention_mask"] = inputs["attention_mask"][:, 1:]
                inputs["input_ids"] = inputs["input_ids"][:, 1:]

        with torch.no_grad():
            outputs = self.model(**inputs)
        if return_inputs:
            return outputs.logits, inputs
        else:
            return outputs.logits
