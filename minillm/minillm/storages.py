import json
import os
import time
from abc import abstractmethod
from typing import Any, Callable, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from .data_types import PPORLElement, PPORLBatch

from utils import get_rank


class BaseRolloutStore(Dataset):
    def __init__(self, capacity=-1):
        self.history: Iterable[Any] = None
        self.capacity = capacity

    @abstractmethod
    def push(self, exps: Iterable[Any]):
        """
        Push experiences to rollout storage
        """
        pass

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    @abstractmethod
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create a dataloader for the rollout store

        :param prep_fn: Applied to RLElement after collation (typically tokenizer)
        :type prep_fn: Callable
        """
        pass
    
    @abstractmethod
    def broadcast(self, batch, src=0, group=None):
        pass
    
    @abstractmethod
    def move_to_device(self, batch, device):
        pass


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, seed):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def save(self, path):
        def exp_to_dict(exp):
            return {k: v for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        
        torch.save(data, os.path.join(path, f"{get_rank()}.pkl"))
            
    def load(self, path):
        data = torch.load(os.path.join(path, f"history_{get_rank()}.pkl"), map_location="cpu")
        self.history = [PPORLElement(**d) for d in data]

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def collate(self, elems: Iterable[PPORLElement]):
        if any([e is None for e in elems]):
            print(elems)
        return PPORLBatch(
            # Left padding of already left-padded queries
            pad_sequence(
                [elem.query_tensor.flip(0) for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            ).flip(1),
            # Right pad the rest, to have a single horizontal query/response split
            pad_sequence(
                [elem.response_tensor for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            ),
            torch.tensor([elem.lens for elem in elems], dtype=torch.long),
            torch.tensor([elem.s_lens for elem in elems], dtype=torch.long),
            pad_sequence(
                [elem.mask for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),            
            pad_sequence(
                [elem.logprobs for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.rewards for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.rev_kl for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.w for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.inf_mask for elem in elems],
                padding_value=0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.t_rewards for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
            pad_sequence(
                [elem.ent_rewards for elem in elems],
                padding_value=0.0,
                batch_first=True,
            ),
        )

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        # sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last)
        # we don't use distributed sampler because the dataset on each device is different
        return DataLoader(
            self, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, generator=self.rng
        )
        
    def broadcast(self, batch: PPORLBatch, src=0, group=None):
        for k, v in batch.__dict__.items():
            dist.broadcast(batch.__dict__[k], src=src, group=group)
            
    def move_to_device(self, batch: PPORLBatch, device):
        for k, v in batch.__dict__.items():
            batch.__dict__[k] = batch.__dict__[k].to(device)