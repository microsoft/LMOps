import random
import torch
import os
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size
from utils import print_rank
from tqdm import tqdm
import json
import numpy as np
from .base_datasets import BaseDataset


class PromptDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1, ada_max_length=False, **kwargs):
        super().__init__(args, tokenizer, split, data_path, num, ada_max_length=ada_max_length, **kwargs)
        self.split_token_id = self.args.split_token_id or len(tokenizer)

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        if (self.epoch, index) < self.skip_offset:
            return None

        if self.order is not None:
            index = int(self.order[self.epoch, index])

        data = self.data[index]
        if self.args.bin_data:
            data = data.astype(int)
            assert self.split_token_id in data, f"Split token {self.split_token_id} not found in data"
            source_len = np.where(data==self.split_token_id)[0][0]
            prompt_ids = data[:source_len]
            response_ids = data[source_len+1:]
        elif self.args.json_data:
            prompt_ids = data["prompt_ids"]
            response_ids = data["output_ids"]

        if len(prompt_ids) + len(response_ids) > self.args.max_length + 1 and self.args.trunc_data:
            if prompt_ids[0] == self.tokenizer.bos_token_id:
                prompt_ids = prompt_ids[1:]
                prompt_ids = prompt_ids[-self.args.max_prompt_length+1:]
                prompt_ids = np.concatenate([np.array([self.tokenizer.bos_token_id]), prompt_ids], axis=0)
            else:
                prompt_ids = prompt_ids[-self.args.max_prompt_length:]
            response_ids = response_ids[:self.args.max_length + 1 - len(prompt_ids)]

        assert len(prompt_ids) + len(response_ids) <= self.args.max_length + 1, \
            f"Prompt and response too long: {len(prompt_ids)} + {len(response_ids)} > {self.args.max_length + 1}"        

        if self.args.remove_bos_in_training:
            assert prompt_ids[0] == self.tokenizer.bos_token_id
            prompt_ids = prompt_ids[1:]

        return index, prompt_ids, response_ids

    def collate(self, samples):
        
        if samples[0] is None:
            return None, None
        
        bs = len(samples)
        if self.ada_max_length:
            max_length = max([len(samp[1]) + len(samp[2]) for samp in samples])
            max_length = min(max_length-1, self.max_length)
        else:
            max_length = self.max_length
        
        model_batch = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, self.max_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "loss_mask": torch.zeros(bs, max_length, dtype=torch.float)
        }
        
        for i, (idx, prompt, response) in enumerate(samples):
            full_ids = np.concatenate([prompt, response], axis=0)
            model_batch["input_ids"][i][:len(full_ids)-1] = torch.tensor(full_ids[:-1], dtype=torch.long)
            model_batch["attention_mask"][i][:len(full_ids)-1] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["label"][i][:len(full_ids)-1] = torch.tensor(full_ids[1:], dtype=torch.long)
            st = max(len(prompt)-1, 0)
            no_model_batch["loss_mask"][i][:len(full_ids)-1] = (torch.tensor(full_ids[:-1], dtype=torch.long) != self.pad_id)
            if not self.args.prompt_data_full_loss:
                no_model_batch["loss_mask"][i][:st] = 0.0
            assert torch.sum(no_model_batch["loss_mask"][i]) > 0, (prompt, response, no_model_batch["loss_mask"][i])
        
        return model_batch, no_model_batch

    def collate_gen(self, samples):
        bs = len(samples)
        
        max_prompt_length = max([len(samp[1]) for samp in samples])
        max_response_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "response_ids": torch.ones(bs, max_response_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, response) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["response_ids"][i][:len(response)] = torch.tensor(response, dtype=torch.long)
        
        return model_batch, no_model_batch