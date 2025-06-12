from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward


class GatingNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim, bias=False))  # for BN
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # Apply the conditional ReLU using the expanded mask
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]
        # return x


# token_pattern = tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
#                                   add_special_tokens=False, )
token_pattern = [128009, 128006, 78191, 128007, 271]


def find_token_for_gating(
    lst,
):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


@dataclass
class CustomOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        hidden_state (`Tuple[torch.FloatTensor]` of length `config.num_hidden_layers`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        prompt_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The embeddings of the prompt tokens.
        gating_output (`torch.FloatTensor` of shape `(batch_size, config.num_objectives)`):
            The logits for the gating network.
        score (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            The final reward score.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Same as score
    """

    reward_quantiles: torch.FloatTensor = None
    rewards: torch.FloatTensor = None
    gating_output: Optional[torch.FloatTensor] = None
    score: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LlamaForRewardModelWithGating31(LlamaPreTrainedModel):
    def __init__(self, config):
        # config = AutoConfig.from_pretrained("nicolinho/QRM-Llama3.1-8B")
        config.torch_dtype = torch.bfloat16
        super().__init__(config)
        self.model = LlamaModel(config)
        self.num_labels = config.num_labels
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 19)
        self.num_quantiles = config.num_quantiles
        self.quantiles = torch.linspace(0.0, 1.0, config.num_quantiles + 2)[1:-1]
        self.regression_layer = nn.Linear(config.hidden_size, config.num_quantiles * self.num_objectives, bias=False)
        self.post_init()
        # Not using torch.eye because it is not supported in BF16
        t = torch.zeros(self.num_objectives, self.num_objectives)
        t[range(self.num_objectives), range(self.num_objectives)] = 1.0
        self.reward_transform_matrix = nn.Parameter(t)
        self.reward_transform_matrix.requires_grad = False

        # Initialize weights and apply final processing
        self.gating = GatingNetwork(
            config.hidden_size,
            config.num_objectives,
            temperature=config_dict.get("gating_temperature", 10),
            hidden_dim=config_dict.get("gating_hidden_dim", 1024),
            n_hidden=config_dict.get("gating_n_hidden", 3),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
            rewards = self.regression_layer(hidden_states.float())
            rewards = rewards.reshape(-1, self.config.num_objectives, self.config.num_quantiles)

            gating_token_positions = [find_token_for_gating(ids.tolist()) for ids in input_ids]
            prompt_embedding = tokens_hidden_states[dummy_iterator, gating_token_positions, :]
            gating_output = self.gating(prompt_embedding.float())

            reward_quantiles_all_adjusted = torch.matmul(
                torch.transpose(rewards.float(), 1, 2), self.reward_transform_matrix
            )
            # [B, num_quantiles, num_objectives]
            reward_quantiles = torch.mul(
                gating_output.unsqueeze(-1).repeat(1, 1, self.num_objectives),
                torch.transpose(reward_quantiles_all_adjusted, 1, 2),
            ).sum(1)

            rewards_expectation = rewards.float().mean(dim=2)

            rewards_expectation_adjusted = rewards_expectation @ self.reward_transform_matrix
            score = torch.sum(gating_output * rewards_expectation_adjusted, dim=1, keepdim=True)

        return CustomOutput(
            reward_quantiles=reward_quantiles,
            rewards=rewards_expectation_adjusted,
            gating_output=gating_output,
            score=score,
            logits=score,
        )


class LlamaForRewardModelWithGating3(LlamaPreTrainedModel):
    def __init__(self, config):
        # config = AutoConfig.from_pretrained("nicolinho/QRM-Llama3-8B")
        config.torch_dtype = torch.bfloat16
        super().__init__(config)
        self.model = LlamaModel(config)
        self.num_labels = config.num_labels
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 19)
        self.num_quantiles = config.num_quantiles
        self.quantiles = torch.linspace(0.0, 1.0, config.num_quantiles + 2)[1:-1]
        self.regression_layer = nn.Linear(config.hidden_size, config.num_quantiles * self.num_objectives, bias=False)
        self.post_init()
        # Not using torch.eye because it is not supported in BF16
        t = torch.zeros(self.num_objectives, self.num_objectives).to(torch.bfloat16)
        t[range(self.num_objectives), range(self.num_objectives)] = 1.0
        self.reward_transform_matrix = nn.Parameter(t)
        self.reward_transform_matrix.requires_grad = False

        # Initialize weights and apply final processing
        self.gating = GatingNetwork(
            config.hidden_size,
            config.num_objectives,
            temperature=config_dict.get("gating_temperature", 10),
            hidden_dim=config_dict.get("gating_hidden_dim", 1024),
            n_hidden=config_dict.get("gating_n_hidden", 3),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, attn_implementation="flash_attention_2"
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        rewards = self.regression_layer(hidden_states.float())
        rewards = rewards.reshape(-1, self.config.num_objectives, self.config.num_quantiles)

        gating_token_positions = [find_token_for_gating(ids.tolist()) for ids in input_ids]
        prompt_embedding = tokens_hidden_states[dummy_iterator, gating_token_positions, :]
        gating_output = self.gating(prompt_embedding.float())

        with torch.autocast(device_type=rewards.device.type, dtype=torch.float32):
            reward_quantiles_all_adjusted = torch.matmul(
                torch.transpose(rewards.float(), 1, 2), self.reward_transform_matrix
            )
            # [B, num_quantiles, num_objectives]
            reward_quantiles = torch.mul(
                gating_output.unsqueeze(-1).repeat(1, 1, self.num_objectives),
                torch.transpose(reward_quantiles_all_adjusted, 1, 2),
            ).sum(1)

        rewards_expectation = rewards.float().mean(dim=2)

        rewards_expectation_adjusted = rewards_expectation @ self.reward_transform_matrix
        score = torch.sum(gating_output * rewards_expectation_adjusted, dim=1, keepdim=True)

        return CustomOutput(
            reward_quantiles=reward_quantiles,
            rewards=rewards_expectation_adjusted,
            gating_output=gating_output,
            score=score,
            logits=score,
        )
