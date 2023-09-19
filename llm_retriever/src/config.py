import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


from logger_config import logger


@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    data_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )

    train_n_passages: int = field(
        default=16,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )
    t: float = field(default=0.01, metadata={"help": "temperature of biencoder training"})
    l2_normalize: bool = field(default=True, metadata={"help": "L2 normalize embeddings or not"})
    full_contrastive_loss: bool = field(default=True, metadata={"help": "use full contrastive loss or not"})

    # used for index search
    do_search: bool = field(default=False, metadata={"help": "run the index search loop"})
    search_split: str = field(default='train', metadata={"help": "which split to search"})
    search_batch_size: int = field(default=128, metadata={"help": "query batch size for index search"})
    search_topk: int = field(default=100, metadata={"help": "return topk search results"})

    # used for knowledge distillation
    do_kd_gen_score: bool = field(default=False, metadata={"help": "run the score generation for distillation"})
    kd_gen_score_split: str = field(default='dev', metadata={
        "help": "Which split to use for generation of teacher score"
    })
    kd_gen_score_batch_size: int = field(default=128, metadata={"help": "batch size for teacher score generation"})

    do_kd_biencoder: bool = field(default=False, metadata={"help": "knowledge distillation to biencoder"})
    kd_cont_loss_weight: float = field(default=0.2, metadata={"help": "weight for contrastive loss"})

    max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(default=10_000, metadata={"help": "max examples for test"})
    freeze_position_embedding: bool = field(
        default=True,
        metadata={'help': 'freeze position embedding'}
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )
    pool_type: str = field(
        default='avg',
        metadata={'help': 'pool type'}
    )
    add_qd_prompt: bool = field(
        default=False,
        metadata={'help': 'add query and document prompt'}
    )

    do_llm_eval: bool = field(default=False, metadata={'help': 'do llm eval'})
    llm_model_name_or_path: str = field(default='huggyllama/llama-7b', metadata={'help': 'llm model name or path'})
    llm_k_shot: int = field(default=8, metadata={'help': 'llm k shot'})
    llm_batch_size_per_device: int = field(default=4, metadata={'help': 'llm batch size'})
    llm_max_input_length: int = field(default=1024, metadata={'help': 'llm max input length'})
    llm_max_decode_length: int = field(default=64, metadata={'help': 'llm max decode length'})
    llm_eval_split: str = field(default='test', metadata={'help': 'llm eval split'})
    llm_eval_tasks: List[str] = field(default_factory=lambda: [])
    llm_constrained_decoding: bool = field(default=True, metadata={'help': 'llm constrained decoding'})

    reward_max_length: int = field(default=384, metadata={"help": "max length for reward model inputs"})
    topk_as_positive: int = field(default=3, metadata={'help': 'candidates with top-k scores as positive'})
    bottomk_as_negative: int = field(default=-1, metadata={'help': 'candidates with bottom-k scores as negative'})
    held_out_tasks: List[str] = field(default_factory=lambda: ['qnli', 'piqa', 'yelp'], metadata={'help': 'held out tasks'})

    def __post_init__(self):
        assert os.path.exists(self.data_dir)
        assert self.pool_type in ['cls', 'avg']

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = self.max_train_samples or 128
            self.num_train_epochs = 1
            self.max_steps = min(self.max_steps, 64)
            self.save_steps = min(self.save_steps, 64)
            self.per_device_train_batch_size = min(2, self.per_device_train_batch_size)
            self.train_n_passages = min(4, self.train_n_passages)
            logger.warning('Dry run: set logging_steps=1')

        if self.do_search:
            if self.model_name_or_path in ['bm25', 'random']:
                self.fp16 = False
                self.local_rank = -1

        if self.do_kd_gen_score:
            assert os.path.exists('{}/{}.jsonl.gz'.format(self.data_dir, self.kd_gen_score_split))

        if torch.cuda.device_count() <= 1:
            self.logging_steps = min(10, self.logging_steps)

        super(Arguments, self).__post_init__()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.label_names = ['labels']
