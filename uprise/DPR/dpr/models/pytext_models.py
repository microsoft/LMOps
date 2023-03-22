#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from pytext.models.representations.transformer_sentence_encoder import TransformerSentenceEncoder
from pytext.optimizer.optimizers import AdamW
from torch import Tensor as T
from torch import nn

from .biencoder import BiEncoder

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(args, inference_only: bool = False):
    # since bert tokenizer is the same in HF and pytext/fairseq, just use HF's implementation here for now
    from .hf_models import get_tokenizer, BertTensorizer

    tokenizer = get_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)

    question_encoder = PytextBertEncoder.init_encoder(args.pretrained_file,
                                                      projection_dim=args.projection_dim, dropout=args.dropout,
                                                      vocab_size=tokenizer.vocab_size,
                                                      padding_idx=tokenizer.pad_token_type_id
                                                      )

    ctx_encoder = PytextBertEncoder.init_encoder(args.pretrained_file,
                                                 projection_dim=args.projection_dim, dropout=args.dropout,
                                                 vocab_size=tokenizer.vocab_size,
                                                 padding_idx=tokenizer.pad_token_type_id
                                                 )

    biencoder = BiEncoder(question_encoder, ctx_encoder)

    optimizer = get_optimizer(biencoder,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = BertTensorizer(tokenizer, args.sequence_length)
    return tensorizer, biencoder, optimizer


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0) -> torch.optim.Optimizer:
    cfg = AdamW.Config()
    cfg.lr = learning_rate
    cfg.weight_decay = weight_decay
    cfg.eps = adam_eps
    optimizer = AdamW.from_config(cfg, model)
    return optimizer


def get_pytext_bert_base_cfg():
    cfg = TransformerSentenceEncoder.Config()
    cfg.embedding_dim = 768
    cfg.ffn_embedding_dim = 3072
    cfg.num_encoder_layers = 12
    cfg.num_attention_heads = 12
    cfg.num_segments = 2
    cfg.use_position_embeddings = True
    cfg.offset_positions_by_padding = True
    cfg.apply_bert_init = True
    cfg.encoder_normalize_before = True
    cfg.activation_fn = "gelu"
    cfg.projection_dim = 0
    cfg.max_seq_len = 512
    cfg.multilingual = False
    cfg.freeze_embeddings = False
    cfg.n_trans_layers_to_freeze = 0
    cfg.use_torchscript = False
    return cfg


class PytextBertEncoder(TransformerSentenceEncoder):

    def __init__(self, config: TransformerSentenceEncoder.Config,
                 padding_idx: int,
                 vocab_size: int,
                 projection_dim: int = 0,
                 *args,
                 **kwarg
                 ):

        TransformerSentenceEncoder.__init__(self, config, False, padding_idx, vocab_size, *args, **kwarg)

        assert config.embedding_dim > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.embedding_dim, projection_dim) if projection_dim != 0 else None

    @classmethod
    def init_encoder(cls, pretrained_file: str = None, projection_dim: int = 0, dropout: float = 0.1,
                     vocab_size: int = 0,
                     padding_idx: int = 0, **kwargs):
        cfg = get_pytext_bert_base_cfg()

        if dropout != 0:
            cfg.dropout = dropout
            cfg.attention_dropout = dropout
            cfg.activation_dropout = dropout

        encoder = cls(cfg, padding_idx, vocab_size, projection_dim, **kwargs)

        if pretrained_file:
            logger.info('Loading pre-trained pytext encoder state from %s', pretrained_file)
            state = torch.load(pretrained_file)
            encoder.load_state_dict(state)
        return encoder

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        pooled_output = super().forward((input_ids, attention_mask, token_type_ids, None))[0]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)

        return None, pooled_output, None

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.representation_dim
