# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty, chunked_full_kl
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from contextlib import contextmanager
import time


if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextmanager
def profile_cuda(name: str, device: str = "cuda", enabled: bool = True):
    if not enabled:
        yield
        return

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    start_time = time.perf_counter()

    try:
        yield
    finally:
        end_event.record()
        torch.cuda.synchronize(device)

        cuda_time_s = start_event.elapsed_time(end_event) * 1e-3
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
        )
        
        total_time_s = time.perf_counter() - start_time

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = torch.distributed.get_rank() == 0
        else:
            is_main = True

        if is_main:
            print(f"[{name}] Total execution time: {total_time_s:.3f} s")
            print(f"[{name}] CUDA execution time: {cuda_time_s:.3f} s")
            print(f"[{name}] CUDA peak memory: {peak_memory_gb:.2f} GB")


def _merge_topk_indices(indices1, indices2, target_k, special_marker=-1):
    """Merge and deduplicate two top-k index tensors on the GPU."""
    combined = torch.cat([indices1, indices2], dim=-1)
    combined_sorted, _ = combined.sort(dim=-1)
    first = torch.ones_like(combined_sorted[..., :1], dtype=torch.bool)
    is_unique = torch.cat(
        [first, combined_sorted[..., 1:] != combined_sorted[..., :-1]],
        dim=-1,
    )
    max_index = combined_sorted.max()
    filler = max_index + 1
    unique_candidates = torch.where(is_unique, combined_sorted, filler)
    result, _ = unique_candidates.sort(dim=-1)
    result = result[..., :target_k]
    return torch.where(result > max_index, special_marker, result)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_all_logits=False, return_hidden_states=False, detach_hidden=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                # Special handling for MiniCPM-o model: pixel_values, image_bound, and tgt_sizes
                # need different concatenation strategies compared to other multimodal inputs
                if (key == "pixel_values" and isinstance(micro_batch["multi_modal_inputs"][0]["pixel_values"], list)) or key == "image_bound" or key == "tgt_sizes":
                    # For MiniCPM-o: keep as list structure instead of concatenating tensors
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if "multi_modal_inputs" in micro_batch:
                    # MiniCPM-o specific processing for image bounds and pixel values
                    if "image_bound" in multi_modal_inputs:
                        # Adjust image bounds based on left padding and cumulative sequence lengths
                        # This is necessary for MiniCPM-o's vision-language alignment
                        left_padding_length = torch.argmax(attention_mask, dim=1)
                        image_bounds = []
                        for i in range(len(multi_modal_inputs["image_bound"])):
                            image_bound = multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
                            image_bounds.append(image_bound)
                        multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
                        # Flatten pixel values list for MiniCPM-o processing
                        pixel_values = []
                        for i in range(len(multi_modal_inputs["pixel_values"])):
                            pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])
                        multi_modal_inputs["pixel_values"] = [pixel_values]
                    # Handle target sizes for MiniCPM-o vision processing
                    if "tgt_sizes" in multi_modal_inputs:
                        multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Register hook to capture lm_head input (hidden states) and weight
                if return_hidden_states:
                    _captured = {}
                    def _hook(module, inp, out):
                        h = inp[0]
                        if detach_hidden:
                            h = h.detach()
                        _captured['h'] = h
                        w = module.weight
                        if hasattr(w, 'full_tensor'):  # DTensor (FSDP v2)
                            w = w.full_tensor()
                        _captured['w'] = w.detach().clone()
                        if not detach_hidden:
                            # Actor update path: also capture weight WITH gradient
                            # clone() survives FSDP resharding and preserves grad connectivity
                            _captured['w_grad'] = w.clone()
                    _handle = self.actor_module.lm_head.register_forward_hook(_hook)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                # Early return for hidden states path (skip logits processing)
                if return_hidden_states:
                    _handle.remove()
                    self._cached_lm_head_weight = _captured['w']
                    if not detach_hidden:
                        self._cached_lm_head_weight_grad = _captured['w_grad']
                    hidden = _captured['h'].squeeze(0)  # (total_nnz, D)
                    hidden_padded = pad_input(
                        hidden_states=hidden,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    hidden_response = hidden_padded[:, -response_length - 1 : -1]  # (bs, R, D)
                    # Compute entropy from logits if needed (for logging)
                    entropy = None
                    if calculate_entropy:
                        logits_rmpad = output.logits.squeeze(0).detach()  # (total_nnz, V)
                        logits_rmpad.div_(temperature)
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        del logits_rmpad  # free (N, V) memory early
                        full_entropy = pad_input(
                            hidden_states=entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                    del output  # free output.logits (N, V) memory
                    return entropy, hidden_response

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    with profile_cuda("logprob_logsoftmax", device=self.device_name, enabled=self.config.profile_kl):
                        if return_all_logits:
                            cache_clear_interval = 0
                            cache_clear_counter = 0
                            # chunk_size = 512
                            chunk_size = logits_rmpad.size(0)
                            log_probs = torch.empty_like(logits_rmpad)
                            for i in range(0, logits_rmpad.size(0), chunk_size):
                                end = i + chunk_size
                                log_probs[i:end] = torch.nn.functional.log_softmax(logits_rmpad[i:end], dim=-1)
                                if is_cuda_available and cache_clear_interval and cache_clear_interval > 0:
                                    cache_clear_counter += 1
                                    if cache_clear_counter % cache_clear_interval == 0:
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                        else:
                            log_probs = logprobs_from_logits(
                                logits=logits_rmpad,
                                labels=input_ids_rmpad_rolled,
                                inplace_backward=inplace_backward,
                            )
                    
                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                topk_already_handled = False
                with profile_cuda("pad_logprob", device=self.device_name, enabled=self.config.profile_kl):
                    if return_all_logits and self.config.kl_topk > 0:
                        # Optimization: handle topk BEFORE pad_input to avoid creating
                        # the huge (bs, seqlen, vocab_size) padded tensor.
                        # Instead we either pad only (N, K) or gather per-sample from rmpad.
                        kl_topk = self.config.kl_topk
                        topk_already_handled = True

                        if "kl_topk_indices" in micro_batch:
                            # Step 2: gather from rmpad log_probs per-sample using cu_seqlens
                            kl_indices = micro_batch["kl_topk_indices"].to(log_probs.device)  # (bs, R, K)
                            K = kl_indices.shape[-1]
                            R = response_length
                            gathered = torch.full((batch_size, R, K), -1e20, dtype=log_probs.dtype, device=log_probs.device)

                            logit_slice_start = seqlen - R - 1
                            for si in range(batch_size):
                                actual_len = (cu_seqlens[si + 1] - cu_seqlens[si]).item()
                                if actual_len <= 0:
                                    continue
                                first_valid_col = indices[cu_seqlens[si]].item() - si * seqlen
                                last_valid_col = first_valid_col + actual_len - 1

                                r_start = max(0, first_valid_col - logit_slice_start)
                                r_end = min(R - 1, last_valid_col - logit_slice_start)
                                n_valid_logits = r_end - r_start + 1
                                if n_valid_logits <= 0:
                                    continue

                                rmpad_offset = (logit_slice_start + r_start) - first_valid_col
                                rmpad_start = cu_seqlens[si].item() + rmpad_offset
                                rmpad_end = rmpad_start + n_valid_logits
                                sample_log_probs = log_probs[rmpad_start:rmpad_end]  # (n_valid_logits, V)

                                sample_kl_indices = kl_indices[si, r_start:r_end + 1]  # (n_valid_logits, K)
                                valid_mask = sample_kl_indices != -1
                                safe_idx = torch.where(valid_mask, sample_kl_indices, torch.zeros_like(sample_kl_indices))
                                sample_gathered = torch.gather(sample_log_probs, -1, safe_idx.long())
                                gathered[si, r_start:r_end + 1] = torch.where(valid_mask, sample_gathered,
                                                                               torch.full_like(sample_gathered, -1e20))

                            log_probs = gathered  # (bs, R, K)

                        elif "first_kl_topk_indices" in micro_batch:
                            # For JSD, combine the actor and reference top-k sets.
                            first_indices = micro_batch["first_kl_topk_indices"].to(log_probs.device)
                            _, current_indices_rmpad = torch.topk(log_probs, k=kl_topk, dim=-1)
                            current_indices_padded = pad_input(
                                hidden_states=current_indices_rmpad.float(),
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            )
                            current_indices = current_indices_padded[
                                :, -response_length - 1 : -1
                            ].long()

                            jsd_beta = self.config.get("jsd_beta", -1)
                            if jsd_beta > 0 and not self.config.get("jsd_full_topk", False):
                                merged_indices = torch.cat(
                                    [first_indices, current_indices],
                                    dim=-1,
                                )
                            else:
                                merged_indices = _merge_topk_indices(
                                    first_indices,
                                    current_indices,
                                    target_k=2 * kl_topk,
                                )
                            log_probs = merged_indices.float()

                        else:
                            # Step 1: topk on rmpad, pad small indices, slice response
                            _, topk_indices_rmpad = torch.topk(log_probs, k=kl_topk, dim=-1)  # (N, K)
                            topk_indices_padded = pad_input(
                                hidden_states=topk_indices_rmpad.float(),
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            )  # (bs, S, K) — tiny
                            log_probs = topk_indices_padded[:, -response_length - 1 : -1]  # (bs, R, K)

                    elif return_all_logits:
                        # kl_topk == 0: must pad full vocab (rare case)
                        full_log_probs = pad_input(
                            hidden_states=log_probs,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        log_probs = full_log_probs[:, -response_length - 1 : -1]
                    else:
                        full_log_probs = pad_input(
                            hidden_states=log_probs.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                topk_already_handled = False
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Register hook to capture lm_head input (hidden states) and weight
                if return_hidden_states:
                    _captured = {}
                    def _hook(module, inp, out):
                        h = inp[0]
                        if detach_hidden:
                            h = h.detach()
                        _captured['h'] = h
                        w = module.weight
                        if hasattr(w, 'full_tensor'):
                            w = w.full_tensor()
                        _captured['w'] = w.detach().clone()
                        if not detach_hidden:
                            _captured['w_grad'] = w.clone()
                    _handle = self.actor_module.lm_head.register_forward_hook(_hook)

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                # Early return for hidden states path
                if return_hidden_states:
                    _handle.remove()
                    self._cached_lm_head_weight = _captured['w']
                    if not detach_hidden:
                        self._cached_lm_head_weight_grad = _captured['w_grad']
                    hidden = _captured['h']  # (bs, seqlen, D)
                    hidden_response = hidden[:, -response_length - 1 : -1]  # (bs, R, D)
                    # Compute entropy from logits if needed (for logging)
                    entropy = None
                    if calculate_entropy:
                        logits = output.logits.detach()
                        logits.div_(temperature)
                        logits = logits[:, -response_length - 1 : -1, :]
                        entropy = verl_F.entropy_from_logits(logits)
                    del output  # free output.logits (N, V) memory
                    return entropy, hidden_response

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if return_all_logits:
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
            # handle topk if needed (skip if already handled in rmpad optimization above)
            if return_all_logits and self.config.kl_topk > 0 and not topk_already_handled:
                if "kl_topk_indices" in micro_batch:
                    # Step 2: gather using provided indices
                    indices = micro_batch["kl_topk_indices"]
                    indices = indices.to(log_probs.device)
                    
                    # Handle special markers (-1) used for padding in merged indices
                    # Create a mask for valid indices (not special markers)
                    valid_mask = indices != -1
                    
                    # Replace special markers with 0 temporarily to avoid index errors
                    safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
                    
                    # Gather logits
                    gathered_log_probs = torch.gather(log_probs, -1, safe_indices.long())
                    
                    # Set special marker positions to a very small log probability
                    # These positions will be masked out during KL computation and won't affect logsumexp
                    log_probs = torch.where(valid_mask, gathered_log_probs, torch.full_like(gathered_log_probs, -1e20))
                elif "first_kl_topk_indices" in micro_batch:
                    # For JSD, combine the actor and reference top-k sets.
                    first_indices = micro_batch["first_kl_topk_indices"].to(log_probs.device)
                    kl_topk = self.config.kl_topk
                    _, current_indices = torch.topk(log_probs, k=kl_topk, dim=-1)

                    jsd_beta = self.config.get("jsd_beta", -1)
                    if jsd_beta > 0 and not self.config.get("jsd_full_topk", False):
                        merged_indices = torch.cat(
                            [first_indices, current_indices],
                            dim=-1,
                        )
                    else:
                        merged_indices = _merge_topk_indices(
                            first_indices,
                            current_indices,
                            target_k=2 * kl_topk,
                        )
                    log_probs = merged_indices.float()

                else:
                    # Step 1: generate topk indices
                    kl_topk = self.config.kl_topk
                    # we only return indices, but we need to return 'log_probs' tensor
                    # so we cast indices to float. calling function must cast back.
                    _, indices = torch.topk(log_probs, k=kl_topk, dim=-1) # (bs, seq, k)
                    log_probs = indices.float()

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        return_all_logits = data.meta_info.get("return_all_logits", False)
        return_hidden_states = data.meta_info.get("return_hidden_states", False)

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            if self.config.kl_loss_type == "full":
                if "kl_topk_indices" in data.batch:
                    select_keys.append("kl_topk_indices")
                if "first_kl_topk_indices" in data.batch:
                    select_keys.append("first_kl_topk_indices")
            
            batch = data.select(batch_keys=select_keys).batch

            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)
        
        log_probs_lst = []
        entropy_lst = []
        with profile_cuda("ref_logprob", device=self.device_name, enabled=self.config.profile_kl):
            for i, micro_batch in enumerate(micro_batches):
                if isinstance(micro_batch, DataProto):
                    micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                
                if self.config.profile_kl:
                    print(f"[DEBUG] Processing micro_batch {i}/{len(micro_batches)}. return_all_logits={return_all_logits}")
                with torch.no_grad():
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, return_all_logits=return_all_logits, return_hidden_states=return_hidden_states)
                if self.config.profile_kl:
                    print(f"[DEBUG] _forward_micro_batch done. log_probs shape: {log_probs.shape}, device: {log_probs.device}")

                log_probs_lst.append(log_probs)
                if calculate_entropy and entropy is not None:
                    entropy_lst.append(entropy)

            log_probs = torch.concat(log_probs_lst, dim=0)
            entropys = None
            if calculate_entropy and entropy_lst:
                entropys = torch.concat(entropy_lst, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]
                if calculate_entropy and entropys is not None:
                    entropys = entropys[revert_indices]
        
        if self.config.profile_kl:
            print(f"[DEBUG] compute_log_prob finished. Shape: {log_probs.shape}")
        
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        stage_merge = data.meta_info.get("stage_merge", False)

        # ref lm_head weight for chunked full-vocab KL (passed via meta_info from fsdp_workers)
        ref_lm_head_weight = data.meta_info.get("ref_lm_head_weight", None)

        if stage_merge:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            use_chunked_full_kl = (self.config.kl_loss_type == "full" and self.config.kl_topk <= 0)
            if use_chunked_full_kl:
                select_keys.append("ref_hidden_states")
            elif self.config.kl_loss_type == "full" and self.config.kl_topk > 0:
                assert "kl_topk_indices" in data.batch
                select_keys.append("kl_topk_indices")
            if self.config.kl_loss_type != "seqkd" and "exp_log_probs" in data.batch:
                select_keys.append("exp_log_probs")
        else:
            use_chunked_full_kl = False
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages"]
            select_keys.append("old_log_probs")
            if "old_entropys" in data.batch:
                select_keys.append("old_entropys")
            if self.config.use_kl_loss:
                select_keys.append("ref_log_prob")
        if multi_turn:
            select_keys.append("loss_mask")
        if "rollout_log_probs" in data.batch:
            select_keys.append("rollout_log_probs")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if self.config.ppo_mini_batch_size > len(batch):
            actual_ppo_mini_batch_size = len(batch)
        else:
            actual_ppo_mini_batch_size = self.config.ppo_mini_batch_size
        
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // actual_ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(actual_ppo_mini_batch_size)

        metrics = {}
        all_entropy_list = []
        all_response_mask_list = []
        save_logprob_dir = data.meta_info.get("save_logprob_dir", "")
        if save_logprob_dir:
            save_logprob_step = data.meta_info.get("global_steps", 0)
            all_logprob_list = []
            all_exp_logprob_list = []
            all_logprob_mask_list = []
            all_adv_list = []
            all_old_logprob_list = []
            all_old_entropy_list = []
            all_entropy_save_list = []
            all_rollout_logprob_list = []
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = actual_ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = actual_ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                cache_clear_interval = 0
                cache_clear_counter = 0
                with profile_cuda("policy_logprob", device=self.device_name, enabled=self.config.profile_kl):
                    for i, data in enumerate(micro_batches):
                        if self.config.profile_kl:
                            print(f"[DEBUG] update_policy Processing micro_batch {i}/{len(micro_batches)}")

                        # Support all hardwares
                        if isinstance(data, DataProto):
                            data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                        elif isinstance(data, dict):
                            for k, v in data.items():
                                if isinstance(v, torch.Tensor):
                                    data[k] = v.to(get_device_id())
                                elif k == "multi_modal_inputs" and v is not None:
                                    data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                                else:
                                    data[k] = v
                        else:
                            data = data.to(get_device_id())  # actor device is cpu when using offload
                        responses = data["responses"]
                        response_length = responses.size(1)
                        attention_mask = data["attention_mask"]
                        if multi_turn:
                            response_mask = data["loss_mask"][:, -response_length:]
                        else:
                            response_mask = attention_mask[:, -response_length:]

                        tis_imp_ratio_cap = self.config.get("tis_imp_ratio_cap", -1)

                        if not stage_merge:
                            advantages = data["advantages"]
                            entropy_coeff = self.config.entropy_coeff
                            loss_agg_mode = self.config.loss_agg_mode

                            # all return: (bsz, response_length)
                            calculate_entropy = True
                            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                            old_log_prob = data["old_log_probs"]
                            clip_ratio = self.config.clip_ratio
                            clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                            clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                            clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

                            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                            if self.config.policy_loss.loss_mode == "vanilla":
                                rollout_lp = data["rollout_log_probs"] if "rollout_log_probs" in data else None
                                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    response_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    clip_ratio_c=clip_ratio_c,
                                    loss_agg_mode=loss_agg_mode,
                                    rollout_log_probs=rollout_lp,
                                    tis_imp_ratio_cap=tis_imp_ratio_cap,
                                )
                            else:
                                policy_loss_fn = get_policy_loss_fn(loss_mode)
                                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config)

                            if entropy_coeff != 0:
                                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                                # compute policy loss
                                policy_loss = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss = pg_loss

                            if self.config.use_kl_loss:
                                ref_log_prob = data["ref_log_prob"]
                                # compute kl loss
                                kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                metrics["actor/kl_loss"] = kl_loss.detach().item()
                                metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        
                        else:
                            loss_agg_mode = self.config.loss_agg_mode
                            use_chunked = use_chunked_full_kl and "ref_hidden_states" in data
                            jsd_beta = self.config.get("jsd_beta", -1)
                            if use_chunked:
                                # Actor-side chunked path: return hidden states WITH gradient
                                entropy, actor_hidden = self._forward_micro_batch(
                                    micro_batch=data, temperature=temperature,
                                    calculate_entropy=True, return_hidden_states=True, detach_hidden=False)
                                log_prob = None  # not needed, KL computed from hidden states
                            else:
                                return_all_logits = self.config.kl_loss_type == "full"
                                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=True, return_all_logits=return_all_logits)

                            exp_log_prob = None if use_chunked else data["exp_log_probs"]

                            with profile_cuda("kl_penalty", device=self.device_name, enabled=self.config.profile_kl):
                                if use_chunked:
                                    device = actor_hidden.device
                                    ref_hidden = data["ref_hidden_states"].to(device)
                                    ref_W = ref_lm_head_weight.to(device)
                                    # Use cached actor lm_head weight WITH gradient (clone from forward hook)
                                    actor_W = self._cached_lm_head_weight_grad.to(device)
                                    kld = chunked_full_kl(
                                        actor_hidden,
                                        actor_W,
                                        ref_hidden,
                                        ref_W,
                                        jsd_beta=jsd_beta,
                                        temperature=temperature,
                                    )
                                else:
                                    kl_type = "jsd" if jsd_beta > 0 else self.config.kl_loss_type
                                    kld = kl_penalty(
                                        logprob=log_prob,
                                        ref_logprob=exp_log_prob,
                                        kl_penalty=kl_type,
                                        kl_renorm_topk=self.config.get("kl_renorm_topk", False),
                                        jsd_beta=jsd_beta,
                                        jsd_full_topk=self.config.get("jsd_full_topk", False),
                                    )
                            if tis_imp_ratio_cap > 0 and "rollout_log_probs" in data:
                                assert self.config.kl_loss_type != "full", \
                                    "TIS is not supported with kl_loss_type=full (log_prob is 3D vocab logprobs, rollout_log_probs is 2D per-token)"
                                tis_ratio = torch.exp(log_prob - data["rollout_log_probs"]).detach()
                                tis_ratio = torch.clamp(tis_ratio, max=tis_imp_ratio_cap)
                                kld = kld * tis_ratio
                            policy_loss = agg_loss(
                                loss_mat=kld,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )

                            if is_cuda_available and cache_clear_interval and cache_clear_interval > 0:
                                cache_clear_counter += 1
                                if cache_clear_counter % cache_clear_interval == 0:
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / actual_ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                        all_entropy_list.append(entropy.detach())
                        all_response_mask_list.append(response_mask.detach())

                        if save_logprob_dir:
                            if log_prob is not None:
                                all_logprob_list.append(log_prob.detach().cpu())
                            if stage_merge and exp_log_prob is not None:
                                all_exp_logprob_list.append(exp_log_prob.detach().cpu())
                            else:
                                all_adv_list.append(advantages.detach().cpu())
                            all_logprob_mask_list.append(response_mask.detach().cpu())
                            if not stage_merge:
                                all_old_logprob_list.append(old_log_prob.detach().cpu())
                            if "old_entropys" in data:
                                all_old_entropy_list.append(data["old_entropys"].detach().cpu())
                            all_entropy_save_list.append(entropy.detach().cpu())
                            if "rollout_log_probs" in data:
                                all_rollout_logprob_list.append(data["rollout_log_probs"].detach().cpu())

                        if not stage_merge:
                            valid_mask = response_mask > 0
                            n_valid = valid_mask.sum().item()
                            if n_valid > 0:
                                valid_adv = advantages[valid_mask]
                                adv_pos_ratio = (valid_adv > 0).float().mean().item()
                                adv_neg_ratio = (valid_adv < 0).float().mean().item()
                            else:
                                adv_pos_ratio = 0.0
                                adv_neg_ratio = 0.0
                            metrics_data = {
                                "actor/pg_loss": pg_loss.detach().item(),
                                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                                "adv_detail/pos_ratio": adv_pos_ratio,
                                "adv_detail/neg_ratio": adv_neg_ratio,
                                "adv_detail/zero_ratio": 1.0 - adv_pos_ratio - adv_neg_ratio,
                            }
                        else:
                            metrics_data = {
                                "actor/policy_loss": policy_loss.detach().item(),
                            }
                        if "rollout_log_probs" in data:
                            rlp = data["rollout_log_probs"]
                            compare_lp = old_log_prob if not stage_merge else log_prob
                            probs_diff = torch.abs(torch.exp(rlp) - torch.exp(compare_lp))
                            probs_diff = torch.masked_select(probs_diff, response_mask.bool())
                            if probs_diff.numel() > 0:
                                metrics_data["rollout_diff/probs_diff_max"] = probs_diff.max().item()
                                metrics_data["rollout_diff/probs_diff_mean"] = probs_diff.mean().item()
                                metrics_data["rollout_diff/probs_diff_std"] = probs_diff.std().item()
                            if tis_imp_ratio_cap > 0:
                                tis_r = torch.exp(compare_lp - rlp)
                                tis_r_masked = torch.masked_select(tis_r, response_mask.bool())
                                if tis_r_masked.numel() > 0:
                                    metrics_data["rollout_diff/tis_ratio_mean"] = tis_r_masked.mean().item()
                                    metrics_data["rollout_diff/tis_ratio_max"] = tis_r_masked.max().item()
                                    metrics_data["rollout_diff/tis_ratio_min"] = tis_r_masked.min().item()
                                    metrics_data["rollout_diff/tis_clipped_frac"] = (tis_r_masked > tis_imp_ratio_cap).float().mean().item()

                        append_to_dict(metrics, metrics_data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        all_entropy = torch.cat(all_entropy_list, dim=0)
        all_response_mask = torch.cat(all_response_mask_list, dim=0)
        entropy_agg = agg_loss(loss_mat=all_entropy, loss_mask=all_response_mask, loss_agg_mode=self.config.loss_agg_mode)
        metrics["actor/entropy"] = [entropy_agg.item()]

        if save_logprob_dir and all_logprob_list:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                import os
                os.makedirs(save_logprob_dir, exist_ok=True)
                save_dict = {
                    "log_prob": torch.cat(all_logprob_list, dim=0),
                    "response_mask": torch.cat(all_logprob_mask_list, dim=0).bool(),
                }
                if all_exp_logprob_list:
                    save_dict["exp_log_prob"] = torch.cat(all_exp_logprob_list, dim=0)
                if all_adv_list:
                    save_dict["advantages"] = torch.cat(all_adv_list, dim=0)
                if all_old_logprob_list:
                    save_dict["old_log_probs"] = torch.cat(all_old_logprob_list, dim=0)
                if all_old_entropy_list:
                    save_dict["old_entropy"] = torch.cat(all_old_entropy_list, dim=0)
                if all_entropy_save_list:
                    save_dict["entropy"] = torch.cat(all_entropy_save_list, dim=0)
                if all_rollout_logprob_list:
                    save_dict["rollout_log_probs"] = torch.cat(all_rollout_logprob_list, dim=0)
                save_path = os.path.join(save_logprob_dir, "logprob.pt")
                torch.save(save_dict, save_path)
                print(f"[SaveLogprob] Saved to {save_path}, shape={save_dict['log_prob'].shape}")

        self.actor_optimizer.zero_grad()
        return metrics
