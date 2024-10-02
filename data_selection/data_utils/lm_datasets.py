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


class LMDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1, ada_max_length=False, **kwargs):
        super().__init__(args, tokenizer, split, data_path, num, ada_max_length=ada_max_length, **kwargs)

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        if (self.epoch, index) < self.skip_offset:
            return None

        if self.order is not None:
            index = int(self.order[self.epoch, index])

        data = self.data[index].astype(int)
        
        if self.args.remove_bos_in_training:
            assert data[0] == self.tokenizer.bos_token_id
            data = data[1:]
    
        return index, data

    def collate(self, samples):
        
        if samples[0] is None:
            return None, None
        
        bs = len(samples)
        if self.ada_max_length:
            max_length = max([len(samp[1]) for samp in samples])
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
        
        for i, (idx, data) in enumerate(samples):
            full_ids = data[:max_length+1]
            model_batch["input_ids"][i][:len(full_ids)-1] = torch.tensor(full_ids[:-1], dtype=torch.long)
            # model_batch["attention_mask"][i][:len(full_ids)-1] = (torch.tensor(full_ids[:-1], dtype=torch.long) != self.pad_id)
            model_batch["attention_mask"][i][:len(full_ids)-1] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["label"][i][:len(full_ids)-1] = torch.tensor(full_ids[1:], dtype=torch.long)
            no_model_batch["loss_mask"][i][:len(full_ids)-1] = (torch.tensor(full_ids[:-1], dtype=torch.long) != self.pad_id)
        
        return model_batch, no_model_batch

    def collate_gen(self, samples):
        bs = len(samples)
        
        max_prompt_length = max([len(samp[1]) for samp in samples])
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch, no_model_batch