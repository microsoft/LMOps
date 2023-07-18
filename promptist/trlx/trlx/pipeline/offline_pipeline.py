from typing import Iterable
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorWithPaddingForMixin:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_ft, features_mixin = [], []
        for feature in features:
            features_ft_single, features_mixin_single = {}, {}
            for key, value in feature.items():
                if "_mixin" in key:
                    features_mixin_single[key.rstrip("_mixin")] = value
                else:
                    features_ft_single[key] = value
            features_ft.append(features_ft_single)
            features_mixin.append(features_mixin_single)

        batch = self.tokenizer.pad(
            features_ft,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_mixin = self.tokenizer.pad(
            features_mixin,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        for key in features_mixin[0].keys():
            batch[key+"_mixin"] = batch_mixin[key]

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Tokenizes texts, and then pads them into batches
    """

    def __init__(self, prompts, aes_score=None, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.mixin = True if len(prompts[0]) == 2 else False

        self.prompts = []
        assert len(prompts) != 0

        for i in range(len(prompts)):
            if isinstance(prompts[i], tuple):
                single_prompt = tokenizer(prompts[i][0])
                mixin_prompt = tokenizer(prompts[i][0], prompts[i][1], return_token_type_ids=True)
                single_prompt["input_ids_mixin"] = mixin_prompt.input_ids
                single_prompt["attention_mask_mixin"] = mixin_prompt.attention_mask
                single_prompt["token_type_ids_mixin"] = mixin_prompt.token_type_ids
            else:
                single_prompt = tokenizer(prompts[i])
            if aes_score is not None:
                single_prompt["aes_score"] = aes_score[i]
            self.prompts.append(single_prompt)

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        if self.mixin:
            collate_fn = (
                DataCollatorWithPaddingForMixin(self.tokenizer) if self.tokenizer else torch.vstack
            )
        else:
            collate_fn = (
                DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
            )
        return DataLoader(
            self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(
        self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones
    ):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        def collate_fn(elems: Iterable[ILQLElement]):
            return ILQLBatch(
                pad_sequence(
                    [x.input_ids for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards for x in elems], batch_first=True, padding_value=0.0
                ),
                pad_sequence(
                    [x.states_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.actions_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.dones for x in elems], batch_first=True, padding_value=0
                ),
            )

        return DataLoader(
            self, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
