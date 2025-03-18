import os

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class Arguments(TrainingArguments):
    max_len: int = field(
        default=3072,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    eval_task: str = field(
        default='hotpotqa',
        metadata={'help': 'evaluation task'}
    )
    eval_split: str = field(
        default='validation',
        metadata={'help': 'evaluation split'}
    )

    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )

    # available: 'forward', 'backward', 'random'
    context_placement: str = field(default='backward', metadata={'help': 'context placement strategy'})
    num_contexts: int = field(default=20, metadata={"help": "number of context passages for each example"})
    num_threads: int = field(default=32, metadata={"help": "number of threads for retriever search"})
    max_path_length: int = field(default=3, metadata={"help": "maximum path length"})
    sample_temperature: float = field(default=0.7, metadata={"help": "temperature for sampling paths"})

    eval_metrics: str = field(default='em_and_f1', metadata={'help': 'evaluation metrics'})
    # available: greedy, tree_search, best_of_n
    decode_strategy: str = field(default='greedy', metadata={'help': 'decoding strategy'})
    best_n: int = field(default=4, metadata={'help': 'best of n'})

    def __post_init__(self):
        super(Arguments, self).__post_init__()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
