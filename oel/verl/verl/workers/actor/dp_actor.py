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
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
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

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_all_logits=False) -> Tuple[torch.Tensor, torch.Tensor]:
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

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

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
                
                with profile_cuda("pad_logprob", device=self.device_name, enabled=self.config.profile_kl):
                    if return_all_logits:
                        full_log_probs = pad_input(
                            hidden_states=log_probs,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    else:
                        full_log_probs = pad_input(
                            hidden_states=log_probs.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if return_all_logits:
                    log_probs = full_log_probs[:, -response_length - 1 : -1]
                else:
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

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
            
            # handle topk if needed
            if return_all_logits and self.config.kl_topk > 0:
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
                    # Merge first indices with current topk (on GPU)
                    # This happens when we need to merge actor and ref indices, or saved and current actor indices
                    # NOTE: by default we do not use this branch
                    first_indices = micro_batch["first_kl_topk_indices"].to(log_probs.device)
                    target_k = 2 * self.config.kl_topk
                    
                    # Compute current topk indices
                    kl_topk = self.config.kl_topk
                    _, current_indices = torch.topk(log_probs, k=kl_topk, dim=-1)  # (bs, seq, k)
                    
                    # Merge on GPU (Vectorized version for efficiency)
                    def merge_topk_indices_gpu(indices1, indices2, target_k, special_marker=-1):
                        combined = torch.cat([indices1, indices2], dim=-1)
                        combined_sorted, _ = combined.sort(dim=-1)

                        shift = torch.full_like(combined_sorted[..., :1], -1) 
                        mask = torch.cat([torch.ones_like(shift, dtype=torch.bool), 
                                        combined_sorted[..., 1:] != combined_sorted[..., :-1]], dim=-1)
                        
                        max_val = combined_sorted.max()
                        filler = max_val + 1
                        unique_candidates = torch.where(mask, combined_sorted, filler)
                        
                        final_sorted, _ = unique_candidates.sort(dim=-1)
                        result = final_sorted[..., :target_k]
                        final_mask = (result > max_val)
                        result[final_mask] = special_marker
                        
                        return result

                    
                    merged_indices = merge_topk_indices_gpu(first_indices, current_indices, target_k, special_marker=-1)
                    
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
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, return_all_logits=return_all_logits)
                if self.config.profile_kl:
                    print(f"[DEBUG] _forward_micro_batch done. log_probs shape: {log_probs.shape}, device: {log_probs.device}")

                log_probs_lst.append(log_probs)
                if calculate_entropy:
                    entropy_lst.append(entropy)

            log_probs = torch.concat(log_probs_lst, dim=0)
            entropys = None
            if calculate_entropy:
                entropys = torch.concat(entropy_lst, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]
                if calculate_entropy:
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
        stage_merge = data.meta_info["stage_merge"]
        on_policy_merge = data.meta_info["on_policy_merge"]

        assert stage_merge
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if self.config.kl_loss_type == "full" and self.config.kl_topk > 0:
            assert "kl_topk_indices" in data.batch
            select_keys.append("kl_topk_indices")
        if self.config.kl_loss_type != "seqkd":
            assert "exp_log_probs" in data.batch
            select_keys.append("exp_log_probs")
        
        if multi_turn:
            select_keys.append("loss_mask")
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

                        loss_agg_mode = self.config.loss_agg_mode
                        return_all_logits = self.config.kl_loss_type == "full"
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=True, return_all_logits=return_all_logits)
                        
                        exp_log_prob = data["exp_log_probs"] if "exp_log_probs" in data else torch.zeros_like(log_prob)
                        with profile_cuda("kl_penalty", device=self.device_name, enabled=self.config.profile_kl):
                            if on_policy_merge:
                                kld = kl_penalty(logprob=log_prob, ref_logprob=exp_log_prob, kl_penalty=self.config.kl_loss_type, kl_renorm_topk=self.config.kl_renorm_topk)
                            else:
                                kld = kl_penalty(logprob=exp_log_prob, ref_logprob=log_prob, kl_penalty=self.config.kl_loss_type, kl_renorm_topk=self.config.kl_renorm_topk)
                        policy_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
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

                        data = {
                            "actor/policy_loss": policy_loss.detach().item(),
                            "actor/entropy": entropy_agg.detach().item(),
                        }
                        
                        append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
