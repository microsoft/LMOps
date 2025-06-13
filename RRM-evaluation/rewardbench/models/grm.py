import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


class ValueHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        # get vhead config, where config from json first
        if hasattr(config, "vhead_layer_type"):
            self.layer_type = config.vhead_layer_type
        else:
            self.layer_type = kwargs.pop("vhead_layer_type", "mlp")
        if hasattr(config, "vhead_num_neurons"):
            num_neurons = config.vhead_num_neurons
        else:
            num_neurons = kwargs.pop("vhead_num_neurons", 1024)
        if hasattr(config, "vhead_num_layers"):
            num_layers = config.vhead_num_layers
        else:
            num_layers = kwargs.pop("vhead_num_layers", 1)

        if self.layer_type == "linear":
            self.summary = nn.Linear(hidden_size, 1)
        else:
            module_lis = []
            input_neurons = hidden_size
            for i in range(num_layers):
                module_lis.extend([nn.Linear(input_neurons, num_neurons), nn.ReLU()])
                input_neurons = num_neurons

            module_lis.append(nn.Linear(num_neurons, 1))
            self.summary = nn.Sequential(*module_lis)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if self.layer_type == "linear" and output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        elif self.layer_type != "linear" and output.dtype != self.summary[0].weight.dtype:
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        return output


class GRewardModel(PreTrainedModel):
    config_class = AutoConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        model = AutoModelForCausalLM.from_config(config)
        self.model = model.model
        self.v_head = ValueHead(self.model.config)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]

        if hasattr(self.v_head.summary, "weight") and last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, "weight") and (
            last_hidden_state.device != self.v_head.summary[0].weight.device
        ):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)

        # use the last token value as reward
        if torch.any(attention_mask[:, 0] == 0):
            # left padding
            last_index = attention_mask.shape[-1] - 1
        else:
            # right padding
            last_index = attention_mask.sum(dim=-1) - 1
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]
        return value


class GRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
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
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
