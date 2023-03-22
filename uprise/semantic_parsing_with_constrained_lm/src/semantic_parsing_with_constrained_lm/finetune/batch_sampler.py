# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections
import random
from typing import Dict, Iterable, List, Optional

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.util import mean
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.datum import FullDatum


class BucketBatchSampler:
    def __init__(
        self,
        batch_size: int,
        bucket_width: int,
        shuffle_buffer_size: Optional[int] = None,
        adaptive: bool = False,
    ):
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle_buffer_size = shuffle_buffer_size
        self.adaptive = adaptive
        self.batch_sizes_generated: List[int] = []

    def batch(self, instances: Iterable[FullDatum]) -> Iterable[List[FullDatum]]:
        # Imagine we create a bunch of buckets based on source sequence length. For each bucket we also compute a target
        # batch size based on its bucket key.
        #
        # E.g., for `batch_size = 1024` and  `bucket_width = 10`:
        #
        # Batch Size: |  1024 |   512 |   341 | ...
        #    Buckets: |  0-10 | 10-20 | 20-30 | ...
        #  Instances: |   x_1 |  x_0  |       | ...
        #             |   x_3 |  x_2  |       | ...
        #             |  ...  |  ...  |  ...  | ...
        #
        # Every time an instance is processed we add it to its corresponding bucket and then check if the current bucket
        # size is equal to the target batch size for that bucket. If it is, then we yield that bucket as the next batch
        # and then empty the bucket.

        # Map from bucket key to list of instances. The bucket key is determined by looking up the length of the
        # sorting key entry in each instance.
        buckets: Dict[int, List[FullDatum]] = collections.defaultdict(list)
        for instance in self._maybe_shuffle(instances):
            key = (len(instance.natural) + len(instance.canonical)) // self.bucket_width
            buckets[key].append(instance)
            if self.adaptive:
                batch_size = max(1, self.batch_size // (key + 1))
            else:
                batch_size = self.batch_size
            bucket = buckets[key]
            if len(bucket) == batch_size:
                buckets[key] = []
                self.batch_sizes_generated.append(len(bucket))
                yield bucket
        for bucket in buckets.values():
            if len(bucket) > 0:
                self.batch_sizes_generated.append(len(bucket))
                yield bucket

    def _maybe_shuffle(self, instances: Iterable[FullDatum]) -> Iterable[FullDatum]:
        if self.shuffle_buffer_size is None:
            for instance in instances:
                yield instance
            return
        instances_iter = iter(instances)
        exhausted = False
        while not exhausted:
            buffer: List[FullDatum] = []
            while len(buffer) < self.shuffle_buffer_size and not exhausted:
                instance = next(instances_iter, None)
                if instance is None:
                    exhausted = True
                else:
                    buffer.append(instance)
            if len(buffer) > 0:
                # TODO: Allow setting the random seed.
                random.shuffle(buffer)
                for instance in buffer:
                    yield instance

    def get_average_batch_size(self) -> float:
        return mean(self.batch_sizes_generated)
