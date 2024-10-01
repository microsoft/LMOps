import random
import torch
import os
from torch.utils.data import Dataset
from torch.distributed import get_rank, get_world_size
from utils import print_rank
from tqdm import tqdm
import json
import numpy as np
import h5py

from .base_datasets import BaseDataset


class DataScorerDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1, ada_max_length=False, **kwargs):
        super().__init__(args, tokenizer, split, data_path, num, ada_max_length=ada_max_length, **kwargs)
        print_rank(f"{self.split}, num {self.num}")
            
    def __len__(self):
        return self.num

    def load_data(self, **kwargs):
        self.data = self.load_data_bin(self.data_path, **kwargs)
        if not self.args.do_infer:
            with h5py.File(os.path.join(self.data_path, f"{self.split}_scores.hdf5"), "r") as f:
                self.scores = f["scores"][:]
                self.scores = self.scores[:self.num]
        else:
            self.scores = None

    def __getitem__(self, index: int):
        if self.scores is None:
            return self.data[index].astype(int), 0, index
        else:
            return self.data[index].astype(int), self.scores[index], index

    def collate(self, samples):
                
        bs = len(samples)
        max_length = self.max_length
        
        model_batch = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            "labels": torch.zeros(bs, dtype=torch.float32),
            "pos": torch.zeros(bs, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long)
        }
        
        for i, (data, score, index) in enumerate(samples):
            model_batch["input_ids"][i][:len(data)] = torch.tensor(data, dtype=torch.long)
            model_batch["attention_mask"][i][:len(data)] = 1
            model_batch["labels"][i] = torch.tensor(score, dtype=torch.float32)
            model_batch["pos"][i] = len(data) - 1
            no_model_batch["idx"][i] = index
        
        return model_batch, no_model_batch
