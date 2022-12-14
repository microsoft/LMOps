import os
import json
import torch
import numpy as np
from argparse import Namespace
import logging

from typing import Any, Dict, Iterator, List
from fairseq import metrics, search, tokenizer, utils

from fairseq.data import (
    data_utils,
    Dictionary,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
    RightPadDataset,
    LeftPadDataset,
    RawLabelDataset,
    FairseqDataset,
    PrependTokenDataset,
    ConcatSentencesDataset,
    AppendTokenDataset,
    PadDataset,
)

from fairseq.data import (
    Dictionary,
    TokenBlockDataset,
    data_utils,
    iterators,
)

from fairseq.tasks import register_task, FairseqDataclass, FairseqTask
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from dataclasses import dataclass, field
from omegaconf import II, MISSING
from typing import Optional
from fairseq import utils

from struprompting.tasks.fewshot_task import CB, BoolQ, COPA, MultiRC, HellaSwag, StoryCloze, Winogrande, Winograd, WiC, WSC, PIQA, OBQA, ARCC, ARCE, SST2, AGNews, SST5, TREC, RTE, IMDB, Subj, DBPedia, MR, NQ, WebQS, TriviaQA, RACEm, RACEh, COQA, SQuADv2, SQuAD

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

logger = logging.getLogger(__name__)

task_map = {
    "cb": CB,
    "boolq": BoolQ,
    "copa": COPA,
    "multirc": MultiRC,
    "hellaswag": HellaSwag,
    "storycloze": StoryCloze,
    "winogrande": Winogrande,
    "winograd": Winograd,
    "wic": WiC,
    "wsc": WSC,
    "piqa": PIQA,
    "obqa": OBQA,
    "arcc": ARCC,
    "arce": ARCE,
    "sst2": SST2,
    "agnews": AGNews,
    "sst5": SST5,
    "trec": TREC,
    "rte": RTE,
    "imdb": IMDB,
    "subj": Subj,
    "dbpedia": DBPedia,
    "mr": MR,
    "nq": NQ,
    "webqs": WebQS,
    "triviaqa": TriviaQA,
    "racem": RACEm,
    "raceh": RACEh,
    "coqa": COQA,
    "sq2": SQuADv2,
    "sq": SQuAD,
}

@dataclass
class FewshotEvalConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    seed: int = II("common.seed")
    eval_data: str = field(default="", metadata={"help": "dataset name"})
    test_split: str = field(default="test", metadata={"help": "test data split"})
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )
    gpt_dict: str = field(
        default="", metadata={"help": "gpt dict file"}
    )
    
    tokens_per_sample: int = field(
        default=2048,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )

    k: int = field(
        default=4,
        metadata={"help": "k shot"},
    )
    temp_index: int = field(
        default=0,
        metadata={"help": "template index"},
    )
    permut_index: int = field(
        default=-1,
        metadata={"help": "permutation index"},
    )


@register_task('fs_eval', dataclass=FewshotEvalConfig)
class FewshotEval(FairseqTask):

    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = dictionary
        self.seed = cfg.seed
        self.tokenizer = tokenizer

        self.fewshot_task = task_map[self.cfg.eval_data](tokenizer=self.tokenizer, dictionary=self.dictionary, k=cfg.k, temp_index=cfg.temp_index, seed=cfg.seed, permut_index=cfg.permut_index)
        self.max_pos_train = 0

        self.src_tokens, self.gpt_loss_mask, self.labels, self.answer_set = self.fewshot_task.get_data_for_fewshot()

        # open-ended generation
        self.generator = None
        self.models = None

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        dictionary = Dictionary.load(cfg.gpt_dict)
        dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)

        while len(dictionary) < 10:
            dictionary = Dictionary.load(cfg.gpt_dict)
            dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)
        logger.info("dictionary: {} types".format(len(dictionary)))

        tokenizer = GPT2BPE(Namespace(
            gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
            gpt2_encoder_json=cfg.gpt2_encoder_json))

        return cls(cfg, dictionary, tokenizer)

    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models

        model = models.build_model(cfg, self, from_checkpoint)
        self.generator = self.build_generator([model], self.cfg)
        self.models = [model]

        return model

    def load_dataset(self, split, combine=False, **kwargs):
        assert split == "train" or split == "valid"
        src_tokens = RawArrayDataset(self.src_tokens)
        gpt_loss_mask = RawArrayDataset(self.gpt_loss_mask, datatype="mask")
        label_ids = RawLabelDataset(self.labels)

        '''
            Input format: src_tokens + option_tokens
        '''
        data_dict = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.dictionary.pad(),
                ),  
                'gpt_loss_mask': RightPadDataset(
                    gpt_loss_mask,
                    pad_idx=False,
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'targets': label_ids,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        dataset = NestedDictionaryDataset(
            data_dict,
            sizes=[src_tokens.sizes],
        )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from struprompting.models.unisequence_generator import (
            UniSequenceGenerator,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            seq_gen_cls = UniSequenceGenerator

        return seq_gen_cls(
            models,
            self.dictionary,
            beam_size=getattr(args, "beam", 3),
            max_len_a=getattr(args, "max_len_a", 1),
            max_len_b=getattr(args, "max_len_b", 30),
            min_len=getattr(args, "min_len", 10),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 0.6),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            external_qkv=False,
            **extra_gen_cls_kwargs,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def valid_step(self, sample, model, criterion, split="valid"):
        assert split == "valid"
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, split=split)
        return loss, sample_size, logging_output


class RawArrayDataset(FairseqDataset):

    def __init__(self, dataset, datatype="token"):
        super().__init__()
        self.dataset = dataset
        self.datatype = datatype
        if hasattr(dataset, 'sizes'):
            self._sizes = dataset.sizes
        else:
            try:
                self._sizes = np.array([len(x) for x in self.dataset])
            except:
                self._sizes =  np.array([1 for x in self.dataset])

    def __getitem__(self, index):
        if type(self.dataset[index][0]) != list:
            if self.datatype == "token":
                return torch.Tensor(self.dataset[index]).long()
            else:
                return torch.Tensor(self.dataset[index]).bool()
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

