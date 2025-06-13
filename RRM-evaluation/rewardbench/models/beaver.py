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
# Mostly copied from:
# https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/models/score_model/llama/modeling_llama.py
# https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/models/score_model/__init__.py
# https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/models/normalizer.py

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.types import Number
from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.llama.modeling_llama import (
    _CONFIG_FOR_DOC,
    LLAMA_INPUTS_DOCSTRING,
)
from transformers.utils.doc import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils.generic import ModelOutput

NormalizeFunction = Literal["affine", "scale", "translate", "identity"]
NormalizerType = Literal["RunningMeanStd", "ExponentialMovingAverage"]


class Normalizer(nn.Module):
    """Normalize input to have zero mean and unit variance."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.LongTensor
    normalize_function: NormalizeFunction

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        if normalize_function not in {"affine", "scale", "translate", "identity"}:
            raise ValueError(
                f"Invalid normalization function type: {normalize_function}. ",
                'Expected one of "affine", "scale", "translate", "identity".',
            )
        self.normalize_function = normalize_function
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long, device=device))

    @abstractmethod
    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return standard deviation."""
        return self.var.sqrt()

    def set_mean_var(
        self,
        mean: torch.Tensor | list[float] | tuple[float, ...] | None,
        var: torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> None:
        """Set mean and variance."""
        mean = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device) if mean is not None else self.mean
        var = torch.as_tensor(var, dtype=self.var.dtype, device=self.var.device) if var is not None else self.var

        assert mean.shape == self.mean.shape
        assert var.shape == self.var.shape

        self.mean = mean
        self.var = var

    def forward(
        self,
        data: torch.Tensor,
        epsilon: Number = 1e-8,
    ) -> torch.Tensor:
        """Update and normalize input."""
        if self.training:
            self.update(data)
        return self.normalize(data, epsilon=epsilon)

    def normalize(
        self,
        data: torch.Tensor,
        epsilon: Number = 1e-8,
    ) -> torch.Tensor:
        """Normalize input."""
        if self.normalize_function == "affine":
            return (data - self.mean.detach()) / (self.std.detach() + epsilon)
        if self.normalize_function == "scale":
            return data / (self.std.detach() + epsilon)
        if self.normalize_function == "translate":
            return data - self.mean.detach()
        if self.normalize_function == "identity":
            return data
        raise ValueError(
            f"Invalid normalization function type: {self.normalize_function}. ",
            'Expected one of "affine", "scale", "translate", "identity".',
        )

    @classmethod
    def instantiate(
        cls,
        normalizer_type: NormalizerType | None,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        **kwargs: Any,
    ):
        """Get a normalizer."""
        if normalizer_type == "RunningMeanStd":
            return RunningMeanStd(
                normalize_function,
                shape=shape,
                device=device,
            )
        if normalizer_type == "ExponentialMovingAverage":
            return ExponentialMovingAverage(
                normalize_function,
                shape=shape,
                device=device,
                **kwargs,
            )
        if normalizer_type is None:
            return IdentityNormalizer(
                normalize_function,
                shape=shape,
                device=device,
            )
        raise ValueError(
            f"Invalid normalization function type: {normalizer_type}. "
            'Expected one of "RunningMeanStd", "ExponentialMovingAverage".',
        )


@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, score_dim)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.
        end_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_dim)`):
            Last hidden state of the sequence at the output of the last layer of the model.
        end_index (`torch.LongTensor` of shape `(batch_size,)`):
            Indices of the end of the sequence.
    """

    scores: torch.FloatTensor | None = None  # size = (B, L, D)
    end_scores: torch.FloatTensor | None = None  # size = (B, D)
    last_hidden_state: torch.FloatTensor | None = None  # size = (B, L, E)
    end_last_hidden_state: torch.FloatTensor | None = None  # size = (B, E)
    end_index: torch.LongTensor | None = None  # size = (B,)


class RunningMeanStd(Normalizer):
    """Running mean and standard deviation."""

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * (self.count * batch_count / total_count)  # pylint: disable=invalid-name
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ExponentialMovingAverage(Normalizer):
    """Exponential moving average."""

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(normalize_function, shape=shape, device=device)
        self.momentum = momentum

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)

        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
        self.count += batch_count  # pylint: disable=no-member


class IdentityNormalizer(Normalizer):
    """Identity normalizer."""

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        self.count += data.size(0)  # pylint: disable=no-member


class ScoreModelMixin:
    """Base class for score models."""

    score_head: nn.Linear
    normalizer: Normalizer
    do_normalize: bool = False
    normalize_function: NormalizeFunction = "affine"
    _is_score_head_initialized: bool = False

    def init_score_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        """Initialize the score head."""
        if self._is_score_head_initialized:
            return

        config.score_dim = kwargs.pop("score_dim", getattr(config, "score_dim", 1))
        config.bias = kwargs.pop("bias", getattr(config, "bias", False))

        config.score_type = kwargs.pop("score_type", getattr(config, "score_type", "reward"))
        if config.score_type == "reward":
            self.normalize_function = "affine"
        elif config.score_type == "cost":
            self.normalize_function = "scale"
        elif config.score_type == "critic":
            self.normalize_function = "identity"
        else:
            raise ValueError(
                f"Invalid score type: {config.score_type}. Expected one of 'reward', 'cost', or 'critic'.",
            )

        config.do_normalize = kwargs.pop(
            "do_normalize",
            getattr(config, "do_normalize", False),
        )
        self.do_normalize = config.do_normalize

        config.normalizer_type = kwargs.pop(
            "normalizer_type",
            getattr(config, "normalizer_type", None),
        )
        if config.normalizer_type not in {"RunningMeanStd", "ExponentialMovingAverage", None}:
            raise ValueError(
                f"Invalid norm type: {config.normalizer_type}."
                "Expected one of 'RunningMeadStd', 'ExponentialMovingAverage', or None.",
            )
        if config.normalizer_type == "ExponentialMovingAverage":
            config.momentum = kwargs.pop("momentum", getattr(config, "momentum", None))
        momentum = getattr(config, "momentum", None)

        self.score_head = nn.Linear(hidden_size, config.score_dim, bias=config.bias)
        self.normalizer = Normalizer.instantiate(
            normalizer_type=config.normalizer_type,
            normalize_function=self.normalize_function,
            shape=(config.score_dim,),
            momentum=momentum,
        )

        mean = getattr(config, "mean", None)
        var = getattr(config, "var", None)
        self.normalizer.set_mean_var(mean, var)

        self._is_score_head_initialized = True

    def get_scores(
        self,
        last_hidden_state: torch.FloatTensor,  # size = (B, L, E)
        attention_mask: torch.BoolTensor | None = None,  # size = (B, L)
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        """Forward pass of the score model."""
        B, L, E = last_hidden_state.size()

        if attention_mask is None:
            if B > 1:
                raise ValueError("'attention_mask' is required when batch size > 1.")
            attention_mask = last_hidden_state.new_ones(B, L, dtype=torch.bool)  # size = (B, L)

        scores = self.score_head(last_hidden_state).float()  # size = (B, L, D)

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # size = (B,)
        end_last_hidden_state = torch.gather(  # size = (B, 1, E)
            last_hidden_state,
            dim=1,
            index=(
                end_index.to(last_hidden_state.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, last_hidden_state.size(-1))
            ),
        )
        end_scores = torch.gather(  # size = (B, 1, D)
            scores,
            dim=1,
            index=(end_index.to(scores.device).unsqueeze(dim=1).unsqueeze(dim=2).expand(-1, -1, scores.size(-1))),
        )
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)  # size = (B, E)
        end_scores = end_scores.squeeze(dim=1)  # size = (B, D)

        if self.training:
            if dist.is_initialized():
                gathered_end_scores_list = [torch.zeros_like(end_scores) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_end_scores_list, end_scores)
                gathered_end_scores = torch.cat(gathered_end_scores_list, dim=0)
                self.normalizer.update(gathered_end_scores)
            else:
                self.normalizer.update(end_scores)
            self.config.mean = self.normalizer.mean.tolist()
            self.config.var = self.normalizer.var.tolist()

        if self.do_normalize:
            scores = self.normalizer.normalize(scores)
            end_scores = self.normalizer.normalize(end_scores)

        if not return_dict:
            return scores, end_scores

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_scores,  # size = (B, D)
            last_hidden_state=last_hidden_state,  # size = (B, L, E)
            end_last_hidden_state=end_last_hidden_state,  # size = (B, E)
            end_index=end_index,  # size = (B,)
        )

    def set_normalize(self, mode: bool = True) -> None:
        if self.do_normalize == mode:
            return

        self.do_normalize = self.config.do_normalize = mode


class LlamaForScore(ScoreModelMixin, LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)

        config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from safe_rlhf.models.score_model.llama.modeling_llama import LlamaForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForScore.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # get score
        >>> outputs = model(**inputs)
        >>> end_scores = outputs.end_scores
        >>> end_scores
        tensor([[0.0000]])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state  # size = (B, L, E)
        return self.get_scores(
            last_hidden_state,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )


# Pipeline addition
class BeaverPipeline:
    # init loads task, tokenizer and model
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
        return outputs.end_scores


# Pipeline addition
class BeaverCostPipeline:
    # init loads task, tokenizer and model
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
        return -outputs.end_scores
