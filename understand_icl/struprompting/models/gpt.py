from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from torch import Tensor

import logging
import json

import math
import torch.onnx.operators
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.utils import safe_getattr, safe_hasattr

from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.transformer_lm import (
  TransformerLanguageModelConfig,
  TransformerLanguageModel,
  base_gpt3_architecture,
)
from fairseq.models.transformer.transformer_decoder import TransformerDecoder


@dataclass
class GPTModelConfig(TransformerLanguageModelConfig):
    gpt_model_path: str = field(
        default="",
        metadata={"help": "gpt checkpoint path"},
    )
    

@register_model("gptmodel", dataclass=GPTModelConfig)
class GPTmodel(TransformerLanguageModel):

    @classmethod
    def build_model(cls, args, task):
        model = TransformerLanguageModel.build_model(args, task)

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_input_dim
        )
        model.decoder = GPTDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )

        if args.gpt_model_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.gpt_model_path)
            model.load_state_dict(state["model"], strict=True, args=args)
        
        # ! ICL analysis
        if task.cfg.ana_setting in ['zs', 'ftzs', 'icl']:
            for p in model.parameters():
                p.requires_grad = False

        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True 
                break
        
        return model


class GPTDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        bos=0,
        pad=1, 
        eos=2
    ):
        self.alpha = None
        super().__init__(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.cfg.decoder.embed_dim,
            self.padding_idx,
            init_size=self.max_target_positions + self.padding_idx + 1,
        )
        self.bos, self.pad, self.eos = bos, pad, eos
        # context examples
        self.context_keys = None
        self.context_values = None
        self.demons_num = 0
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        external_qkv: bool=False,
        **kwargs
    ):
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            attention_mask=attention_mask,
            positions=positions,
            external_qkv=external_qkv,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        external_qkv: bool=False,
    ):
        """
        Similar to *extract_features_scriptable* in transformer_decoder.py
        but use contexualized features as input
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state, external_qkv=external_qkv
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        gpt_embed_output = self.embed_tokens(prev_output_tokens)

        x = self.embed_scale * gpt_embed_output

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        qkv_val = []

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        self_attn_out_hiddens: List[Optional[Tensor]] = []

        for idx, layer in enumerate(self.layers):
            if attention_mask is None:
                if incremental_state is None and not full_context_alignment:
                    self_attn_mask = self.buffered_future_mask(x)
                else:
                    self_attn_mask = None
            else:
                self_attn_mask = attention_mask

            layer_attn = None
            x, qkv_val_sub, _, self_attn_out_hidden = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                key_val=self.context_keys[idx] if external_qkv else None,
                value_val=self.context_values[idx] if external_qkv else None,
                demons_num=self.demons_num,
            )
            qkv_val.append(qkv_val_sub)
            self_attn_out_hiddens.append(self_attn_out_hidden)

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if self.alpha is not None:
                x = torch.mul(self.alpha, x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "qkv_val":qkv_val, "self_attn_out_hiddens":self_attn_out_hiddens}

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False, max_pos=None, external_qkv=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.

    assert max_pos is not None
    mask = tensor.ne(padding_idx).int()

    if external_qkv == False and max_pos > 0:
        # left padding 
        pos = torch.cumsum(mask, dim=1).long()
        max_pos_cur = pos.max(-1)[0]
        pos += (max_pos - max_pos_cur + 1).unsqueeze(-1)
    else:
        mask = tensor.ne(padding_idx).int()
        pos = torch.cumsum(mask, dim=1) + max_pos
    pos = (pos.type_as(mask) * mask).long() + padding_idx
    pos[:,0] = padding_idx + 1
    return pos

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)
        self.max_pos_train = 0

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        external_qkv=False
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace, external_qkv=external_qkv, max_pos=self.max_pos_train
        )

        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


@register_model_architecture("gptmodel", "gptmodel_small")
def gptmodel_large(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    base_gpt3_architecture(args)


@register_model_architecture("gptmodel", "gptmodel_large")
def gptmodel_large(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    base_gpt3_architecture(args)

@register_model_architecture("gptmodel", "gptmodel_xl")
def gptmodel_xl(args):
    # 2.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    base_gpt3_architecture(args)

@register_model_architecture("gptmodel", "gptmodel_xxl")
def gptmodel_xl(args):
    # 6.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    base_gpt3_architecture(args)

@register_model_architecture("gptmodel", "gptmodel_huge")
def gptmodel_huge(args):
    # 13B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 40)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    base_gpt3_architecture(args)
