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
"""Same-instance and iterative L2C evaluation flows."""

import json
import os
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import compute_reward


class L2CEvaluationMixin:
    def _l2l_dump_dir(self, subdir):
        """Resolve ${dump_dir}/${exp_name}/${subdir}. Returns None if dumping
        is disabled. Creates the directory on first call."""
        base = self.config.trainer.get("dump_dir", None)
        if not base:
            return None
        exp_name = self.config.trainer.experiment_name
        path = os.path.join(base, exp_name, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def _dump_jsonl(self, path, records, append=False):
        """Write a list of dicts as one JSON object per line."""
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _decode_prompt_full(self, prompt_ids):
        """Decode a left-padded prompt id tensor to the literal string the
        model saw — chat-template special tokens kept, pad tokens stripped.
        Use for rollout dumps so reviewers see exactly what was fed in,
        including <|im_start|>user / system / assistant scaffolding."""
        pad_id = self.tokenizer.pad_token_id
        ids = prompt_ids[prompt_ids != pad_id]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _decode_prompts_batch(self, prompt_ids_batch):
        """Vectorized helper: decode a (B, L) prompt tensor → list[str]."""
        return [self._decode_prompt_full(prompt_ids_batch[i]) for i in range(len(prompt_ids_batch))]

    def _compute_binary_reward_math(
        self,
        experiences,
        exp_batch,
        source_raw_prompts,
        n,
        return_dump_info=False,
        compute_log_prob=False,
    ):
        """
        Re-solve each source problem with the candidate experience prepended,
        then score with self.reward_fn (math_verify) → 0/1 binary reward.

        Replaces the old NLL-as-reward path. Each candidate exp_i is paired
        with source problem (i // n); we run a single actor.generate per
        candidate, then ask reward_fn whether the produced solution is
        correct. The scalar 0/1 reward feeds GRPO group-normalization
        downstream — no explicit baseline needed (group-mean centering
        handles it).

        Args:
            experiences: list[str] of length B*n. Order matches Phase A's
                exp_output (n contiguous candidates per source problem).
            exp_batch: DataProto with B source problems. Carries non_tensor
                'reward_model' / 'data_source' (untouched by Phase A pop).
            source_raw_prompts: list/ndarray of length B holding the chat-format
                msgs ([{"role": "user", "content": ...}]) for each source
                problem. Caller must capture this BEFORE the Phase A pop —
                pop steals 'raw_prompt' out of exp_batch.non_tensor_batch.
            n: rollout width (number of candidates per source problem).
            compute_log_prob: if true, return per-rollout response-token NLL
                and length diagnostics after an extra actor forward pass.
        Returns:
            reward_tensor: shape (B*n,) float, values in {0.0, 1.0}.
                Note: reward_fn (math_dapo) actually returns {-1.0, +1.0};
                we convert to 0/1 accuracy here. GRPO is affine-invariant
                so this conversion does not change advantages.
        """
        EXPERIENCE_SOLVE_PROMPT_TEMPLATE = self.experience_solve_prompt_template
        B = len(exp_batch)
        assert len(experiences) == B * n, \
            f"experiences len {len(experiences)} != B*n = {B}*{n}"
        assert len(source_raw_prompts) == B, \
            f"source_raw_prompts len {len(source_raw_prompts)} != B = {B}"

        # Build B*n re-solve prompts: candidate i → source problem i // n
        tokenized_list = []
        for i, exp in enumerate(experiences):
            src_idx = i // n
            msgs = deepcopy(source_raw_prompts[src_idx])
            problem_text = msgs[-1]['content']
            if exp and exp != "No previous experience.":
                updated_content = EXPERIENCE_SOLVE_PROMPT_TEMPLATE.format(
                    experience=exp, prompt=problem_text
                )
            else:
                updated_content = problem_text
            msgs[-1]['content'] = updated_content
            tokenized_list.append(self.train_dataset.re_tokenize(msgs))

        # Tensor-only batch for generation. Kept separate from the reward
        # metadata batch so resolve_out (which carries its own input_ids /
        # attention_mask / position_ids covering prompt+response) doesn't
        # collide on union later. Mirrors the Phase A pop pattern.
        gen_batch = DataProto.from_single_dict({
            "input_ids": torch.stack([t["input_ids"] for t in tokenized_list]),
            "attention_mask": torch.stack([t["attention_mask"] for t in tokenized_list]),
            "position_ids": torch.stack([t["position_ids"] for t in tokenized_list]),
        })
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
        }

        gen_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, self.actor_rollout_wg.world_size
        )
        gen_padded.meta_info["n"] = 1
        resolve_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        resolve_out = unpad_dataproto(resolve_out_padded, pad_size)

        # Attach reward_model + data_source for each candidate by indexing
        # back into the source problems (i → i // n). compute_reward reads
        # these out of non_tensor_batch to score each solve against the
        # right ground truth.
        src_indices = np.array([i // n for i in range(B * n)])
        for key in ("reward_model", "data_source"):
            if key in exp_batch.non_tensor_batch:
                resolve_out.non_tensor_batch[key] = exp_batch.non_tensor_batch[key][src_indices]

        reward_tensor_full, _ = compute_reward(resolve_out, self.reward_fn)
        # math_dapo (and most math reward fns in verl) return +1.0 for correct
        # and -1.0 for incorrect — NOT {0, 1}. Convert to 0/1 accuracy so that
        # (a) the metric `reward/mean_correct` reads as a true accuracy, and
        # (b) it matches Phase A's `train_acc = sum(r == 1.0) / len` convention.
        # GRPO is affine-invariant in reward, so {0, 1} and {-1, +1} give the
        # same advantages → no gradient change.
        per_sample_raw = reward_tensor_full.sum(-1).float()       # ∈ {-1, +1}
        per_sample_acc = (per_sample_raw == 1.0).float()          # ∈ {0, 1}

        print(f"[Binary Reward] B={B} n={n} acc={per_sample_acc.mean().item():.4f} raw_reward_mean={per_sample_raw.mean().item():.4f}")

        per_rollout_stats = None
        if compute_log_prob:
            resolve_out.meta_info["recompute_log_prob"] = True
            log_prob_padded, log_prob_pad = pad_dataproto_to_divisor(
                resolve_out, self.actor_rollout_wg.world_size
            )
            log_prob_output_padded = self.actor_rollout_wg.compute_log_prob(
                log_prob_padded
            )
            log_prob_output = unpad_dataproto(log_prob_output_padded, log_prob_pad)
            response_length = resolve_out.batch["responses"].shape[-1]
            response_mask = resolve_out.batch["attention_mask"][
                :, -response_length:
            ].float()
            negative_log_prob = -log_prob_output.batch["old_log_probs"].float()
            nll_sum = (negative_log_prob * response_mask).sum(dim=-1)
            valid_length = response_mask.sum(dim=-1)
            per_rollout_stats = {
                "nll_mean": nll_sum / valid_length.clamp_min(1.0),
                "nll_sum": nll_sum,
                "resp_len": valid_length,
            }

        if return_dump_info:
            resolve_responses = [
                self.tokenizer.decode(resolve_out.batch["responses"][i], skip_special_tokens=True)
                for i in range(B * n)
            ]
            # Decode the full chat-templated prompt the model actually saw
            # (after re_tokenize + chat template wrapping). Strip pad tokens
            # but keep <|im_start|> / role markers so reviewers see verbatim.
            resolve_full_prompts = self._decode_prompts_batch(resolve_out.batch["prompts"])
            return (
                per_sample_acc,
                resolve_responses,
                resolve_full_prompts,
                per_rollout_stats,
            )
        return per_sample_acc

    def _parse_experience(self, exp_text):
        """Parse exp_learner output into a stored experience string. Mirrors
        the Phase A parsing in fit() so eval and train interpret candidates
        the same way. Returns "" if parsing yields nothing usable."""
        if "</think>" in exp_text:
            exp_text = exp_text.split("</think>")[-1]
        if self.config.trainer.prompt_version == 'v1':
            marker = "- EXPERIENCE ITEM:"
            lines = [l for l in exp_text.split("\n") if marker in l]
            result_lines = []
            for l in lines:
                parts = l.split(marker)[1:]
                result_lines.extend([marker + p.rstrip() for p in parts if p.strip()])
            return "\n".join(result_lines)
        elif self.config.trainer.prompt_version == 'v2':
            lines = [l.strip() for l in exp_text.split("\n") if l.strip().startswith("- ")]
            return "\n".join(lines)
        else:
            return exp_text.strip()

    def _run_l2l_eval(self, logger):
        """L2L evaluation aligned with training Phase A/B/C.

        For each val problem (B per batch, exhausting val_dataloader up to
        ``trainer.eval_max_problems``):
          1. Phase A — actor solves with NO experience → 0/1 acc_no_exp
          2. exp_learner generates n candidate experiences from
             (problem, Phase A solution), exactly like training Phase A
             with max_exp_steps=1 (PREVIOUS_EXPERIENCE = "No previous experience.")
          3. Phase C — actor re-solves the source problem once per
             candidate (B*n re-solves total) → 0/1 acc per candidate

        Aggregated metrics (logged at the loaded ckpt's global_step):
          - eval/acc_no_exp           : baseline (no experience) accuracy
          - eval/acc_with_exp_mean    : mean per-candidate accuracy
          - eval/acc_best_of_n        : fraction of problems where ANY of n
                                        candidate experiences solves it
          - eval/delta_with_exp_mean  : with_exp_mean − no_exp
          - eval/delta_best_of_n      : best_of_n   − no_exp
          - eval/parse_success_rate   : fraction of candidates that parse

        Reuses _compute_binary_reward_math for Phase C so the grading path
        is byte-identical to training.
        """
        EXPERIENCE_UPDATE_PROMPT = self.experience_update_prompt
        n = self.config.actor_rollout_ref.rollout.n
        eval_max_problems = int(self.config.trainer.get("eval_max_problems", 500))
        compute_loss = bool(
            OmegaConf.select(
                self.config.trainer, "compute_loss_metrics", default=False
            )
        )

        def _new_loss_accumulator():
            return {"nll_mean": 0.0, "nll_sum": 0.0, "resp_len": 0.0, "count": 0}

        def _accumulate_loss(accumulator, stats):
            if stats is None:
                return
            count = int(stats["nll_mean"].numel())
            accumulator["nll_mean"] += stats["nll_mean"].sum().item()
            accumulator["nll_sum"] += stats["nll_sum"].sum().item()
            accumulator["resp_len"] += stats["resp_len"].sum().item()
            accumulator["count"] += count

        loss_no_exp = _new_loss_accumulator()
        loss_with_exp = _new_loss_accumulator()

        # Phase A dump-only mode: produce a no_exp.jsonl cache and exit (skip
        # Phase B/C). Used by dump_phase_a.yaml to build the actor-baseline
        # cache that subsequent evals consume via phase_a_cache_path.
        phase_a_dump_only = bool(OmegaConf.select(self.config.trainer, "phase_a_dump_only", default=False))
        phase_a_cache_output = OmegaConf.select(self.config.trainer, "phase_a_cache_output_path", default=None)

        # Dump dir for this eval pass (one per loaded ckpt)
        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval")
        no_exp_path = os.path.join(eval_dump_dir, "no_exp.jsonl") if eval_dump_dir else None
        exp_gen_path = os.path.join(eval_dump_dir, "exp_gen.jsonl") if eval_dump_dir else None
        with_exp_path = os.path.join(eval_dump_dir, "with_exp_resolve.jsonl") if eval_dump_dir else None
        # Truncate any leftover from a prior partial run.
        for p in (no_exp_path, exp_gen_path, with_exp_path):
            if p and os.path.exists(p):
                os.remove(p)

        no_exp_correct = 0.0
        with_exp_correct = 0.0
        best_of_n_correct = 0.0
        parse_success_count = 0
        n_problems = 0
        n_candidates = 0

        val_iter = iter(self.val_dataloader)
        batch_idx = 0

        # ── Phase A cache (optional): skip actor.generate at Phase A and use
        # pre-computed (problem, rollout) -> response pairs. Lets multi-step
        # eval share a single Phase A across runs, eliminating sampling noise
        # in acc_no_exp and the cross-step delta. Built by tools/dump_phase_a_cache.py
        _phase_a_cache_path = OmegaConf.select(self.config.trainer, "phase_a_cache_path", default=None)
        _phase_a_cache: dict = {}
        if _phase_a_cache_path:
            with open(_phase_a_cache_path, "r", encoding="utf-8") as _cf:
                for _line in _cf:
                    _r = json.loads(_line)
                    _phase_a_cache.setdefault(_r["global_problem_idx"], []).append(_r)
            for _gpi in _phase_a_cache:
                _phase_a_cache[_gpi].sort(key=lambda x: x["rollout_idx"])
            print(f"[L2L Eval] Phase A cache loaded: {len(_phase_a_cache)} problems from {_phase_a_cache_path}")
        if compute_loss and _phase_a_cache:
            print(
                "[L2L Eval] WARNING: Phase A cache skips the actor forward, so "
                "only with-experience NLL/length metrics will be available"
            )

        while n_problems < eval_max_problems:
            try:
                batch_dict = next(val_iter)
            except StopIteration:
                print(f"[L2L Eval] val dataloader exhausted at {n_problems}/{eval_max_problems}")
                break

            batch = DataProto.from_single_dict(batch_dict)
            # Trim the last batch if it would push us past eval_max_problems
            # so the eval set is exactly eval_max_problems samples regardless
            # of val_batch_size (otherwise we round up to the next multiple).
            remaining = eval_max_problems - n_problems
            if len(batch) > remaining:
                batch = batch[:remaining]
            B = len(batch)

            # Snapshot raw_prompt before pop — re-solve in Phase C needs it
            source_raw_prompts = list(batch.non_tensor_batch['raw_prompt'])

            if _phase_a_cache:
                # ── Cache path: Phase A skipped, load responses + acc from file ──
                _cache_recs = []
                for _bi in range(B):
                    _gpi = n_problems + _bi
                    _recs = _phase_a_cache.get(_gpi, [])
                    if len(_recs) < n:
                        raise ValueError(
                            f"Phase A cache missing rollouts for global_problem_idx={_gpi}: "
                            f"need {n}, have {len(_recs)}"
                        )
                    _cache_recs.extend(_recs[:n])
                phase_a_responses = [_r["response_text"] for _r in _cache_recs]
                phase_a_acc = torch.tensor([float(_r["correct"]) for _r in _cache_recs])
                no_exp_correct += phase_a_acc.sum().item()
                if no_exp_path is not None:
                    try:
                        # Overwrite global_problem_idx + batch_idx for this run's dump
                        # while preserving everything else from the cache record.
                        recs = []
                        for i, _r in enumerate(_cache_recs):
                            rec = dict(_r)
                            rec["batch_idx"] = batch_idx
                            rec["problem_idx_in_batch"] = i // n
                            rec["rollout_idx"] = i % n
                            rec["global_problem_idx"] = n_problems + (i // n)
                            recs.append(rec)
                        self._dump_jsonl(no_exp_path, recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Dump] no_exp cache copy failed: {e}")
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
                )
                # Eval Phase A: no experience injected, so dataloader's tokenization
                # is already the right plain-problem prompt. Just clean non_tensor
                # the same way training Phase A does before generate_sequences.
                gen_batch.non_tensor_batch.pop("raw_prompt_ids", None)
                gen_batch.non_tensor_batch.pop("raw_prompt", None)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": True,
                }
                gen_padded, pad_size = pad_dataproto_to_divisor(
                    gen_batch, self.actor_rollout_wg.world_size
                )
                # n = rollouts per problem in eval (variance reduction).
                # Phase A actor samples n responses per problem; each response
                # feeds exp_learner (n=1) → 1 candidate experience per response;
                # Phase C re-solves each candidate once. So each source problem
                # contributes n acc_no_exp samples + n with_exp samples.
                gen_padded.meta_info["n"] = n
                gen_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                gen_out = unpad_dataproto(gen_out_padded, pad_size * n)

                if compute_loss:
                    gen_out.meta_info["recompute_log_prob"] = True
                    log_prob_padded, log_prob_pad = pad_dataproto_to_divisor(
                        gen_out, self.actor_rollout_wg.world_size
                    )
                    log_prob_output_padded = self.actor_rollout_wg.compute_log_prob(
                        log_prob_padded
                    )
                    log_prob_output = unpad_dataproto(
                        log_prob_output_padded, log_prob_pad
                    )
                    response_length = gen_out.batch["responses"].shape[-1]
                    response_mask = gen_out.batch["attention_mask"][
                        :, -response_length:
                    ].float()
                    negative_log_prob = -log_prob_output.batch[
                        "old_log_probs"
                    ].float()
                    nll_sum = (negative_log_prob * response_mask).sum(dim=-1)
                    valid_length = response_mask.sum(dim=-1)
                    _accumulate_loss(
                        loss_no_exp,
                        {
                            "nll_mean": nll_sum / valid_length.clamp_min(1.0),
                            "nll_sum": nll_sum,
                            "resp_len": valid_length,
                        },
                    )

                # Phase A reward → no-experience baseline accuracy.
                # gen_out now has B*n rows; cannot union with B-sized batch.
                # Attach metadata by repeating each source's reward_model /
                # data_source n times (mirrors _compute_binary_reward_math).
                src_indices = np.array([i // n for i in range(B * n)])
                for key in ("reward_model", "data_source"):
                    if key in batch.non_tensor_batch:
                        gen_out.non_tensor_batch[key] = batch.non_tensor_batch[key][src_indices]
                phase_a_reward, _ = compute_reward(gen_out, self.reward_fn)
                phase_a_acc = (phase_a_reward.sum(-1) == 1.0).float()  # (B*n,)
                no_exp_correct += phase_a_acc.sum().item()

                # ── Phase B: exp_learner generates 1 candidate per Phase A response ──
                phase_a_responses = [
                    self.tokenizer.decode(gen_out.batch["responses"][i], skip_special_tokens=True)
                    for i in range(B * n)
                ]

                # Phase A dump (per batch, append). One row per Phase A rollout,
                # so B*n rows. Adds rollout_idx field to disambiguate which of
                # the n rollouts a given row corresponds to.
                if no_exp_path is not None:
                    try:
                        rm = batch.non_tensor_batch.get("reward_model", None)
                        ds = batch.non_tensor_batch.get("data_source", None)
                        phase_a_acc_list = phase_a_acc.cpu().tolist()
                        phase_a_full_prompts = self._decode_prompts_batch(gen_out.batch["prompts"])
                        recs = []
                        for i in range(B * n):
                            src_idx = i // n
                            rollout_idx = i % n
                            gt = rm[src_idx].get("ground_truth") if rm is not None else None
                            recs.append({
                                "batch_idx": batch_idx,
                                "problem_idx_in_batch": src_idx,
                                "rollout_idx": rollout_idx,
                                "global_problem_idx": n_problems + src_idx,
                                "prompt_text": source_raw_prompts[src_idx][-1]["content"],
                                "full_prompt": phase_a_full_prompts[i],
                                "response_text": phase_a_responses[i],
                                "ground_truth": gt,
                                "data_source": str(ds[src_idx]) if ds is not None else None,
                                "correct": bool(phase_a_acc_list[i] == 1.0),
                            })
                        self._dump_jsonl(no_exp_path, recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Dump] no_exp dump failed: {e}")
            if phase_a_dump_only:
                # Phase A only: skip Phase B/C, advance counters, next batch.
                n_problems += B
                batch_idx += 1
                continue
            exp_learner_prompts = []
            for i in range(B * n):
                src_idx = i // n
                latest_experience = f"Input: {source_raw_prompts[src_idx][-1]['content']}\nOutput: {phase_a_responses[i]}"
                exp_learner_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                    PREVIOUS_EXPERIENCE="No previous experience.",
                    LATEST_EXPERIENCE=latest_experience,
                ))

            exp_tokenized = [self.train_dataset.re_tokenize([{"role": "user", "content": p}])
                             for p in exp_learner_prompts]
            exp_learner_batch = DataProto.from_single_dict({
                "input_ids": torch.stack([t["input_ids"] for t in exp_tokenized]),
                "attention_mask": torch.stack([t["attention_mask"] for t in exp_tokenized]),
                "position_ids": torch.stack([t["position_ids"] for t in exp_tokenized]),
            })
            exp_learner_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "n": 1,
            }
            exp_padded, exp_pad_size = pad_dataproto_to_divisor(
                exp_learner_batch, self.exp_learner_wg.world_size
            )
            exp_out_padded = self.exp_learner_wg.generate_sequences(exp_padded)
            exp_out = unpad_dataproto(exp_out_padded, exp_pad_size)

            experiences = []
            raw_exp_texts = []
            parsed_exps = []
            parse_oks = []
            for i in range(B * n):
                exp_text = self.tokenizer.decode(exp_out.batch["responses"][i], skip_special_tokens=True)
                raw_exp_texts.append(exp_text)
                parsed = self._parse_experience(exp_text)
                parsed_exps.append(parsed)
                parse_oks.append(bool(parsed))
                if parsed:
                    parse_success_count += 1
                # Mirror training (max_exp_steps=1 path): no PREVIOUS to combine
                # with, so combined = parsed if any, else fall back to baseline.
                combined = parsed if parsed else "No previous experience."
                combined = self._truncate_experience(combined, self.config.trainer.experience_max_length)
                experiences.append(combined)

            # Phase B exp_gen dump (per batch, append)
            if exp_gen_path is not None:
                try:
                    exp_gen_full_prompts = self._decode_prompts_batch(exp_out.batch["prompts"])
                    recs = []
                    for i in range(B * n):
                        src_idx = i // n
                        rollout_idx = i % n
                        recs.append({
                            "batch_idx": batch_idx,
                            "candidate_idx_in_batch": i,
                            "source_idx_in_batch": src_idx,
                            "rollout_idx": rollout_idx,
                            "global_source_idx": n_problems + src_idx,
                            "exp_learner_input": exp_learner_prompts[i],
                            "full_prompt": exp_gen_full_prompts[i],
                            "raw_output": raw_exp_texts[i],
                            "parsed_exp": parsed_exps[i],
                            "parse_ok": parse_oks[i],
                            "combined_truncated": experiences[i],
                        })
                    self._dump_jsonl(exp_gen_path, recs, append=True)
                except Exception as e:
                    print(f"[L2L Eval Dump] exp_gen dump failed: {e}")

            # ── Phase C: re-solve once per candidate; reuse training scorer ──
            (
                per_candidate_acc,
                resolve_responses_text,
                resolve_full_prompts,
                phase_c_stats,
            ) = self._compute_binary_reward_math(
                experiences, batch, source_raw_prompts, n,
                return_dump_info=True,
                compute_log_prob=compute_loss,
            )  # (B*n,) ∈ {0, 1}
            _accumulate_loss(loss_with_exp, phase_c_stats)

            # Phase C dump (per batch, append)
            if with_exp_path is not None:
                try:
                    rm = batch.non_tensor_batch.get("reward_model", None)
                    ds = batch.non_tensor_batch.get("data_source", None)
                    rewards_list = per_candidate_acc.cpu().tolist()
                    recs = []
                    for i in range(B * n):
                        src_idx = i // n
                        rollout_idx = i % n
                        gt = rm[src_idx].get("ground_truth") if rm is not None else None
                        recs.append({
                            "batch_idx": batch_idx,
                            "candidate_idx_in_batch": i,
                            "source_idx_in_batch": src_idx,
                            "rollout_idx": rollout_idx,
                            "global_source_idx": n_problems + src_idx,
                            "experience_used": experiences[i],
                            "problem_text": source_raw_prompts[src_idx][-1]["content"],
                            "full_prompt": resolve_full_prompts[i],
                            "response_text": resolve_responses_text[i],
                            "reward": float(rewards_list[i]),
                            "ground_truth": gt,
                            "data_source": str(ds[src_idx]) if ds is not None else None,
                        })
                    self._dump_jsonl(with_exp_path, recs, append=True)
                except Exception as e:
                    print(f"[L2L Eval Dump] with_exp dump failed: {e}")

            with_exp_correct += per_candidate_acc.sum().item()
            n_candidates += B * n

            cand_matrix = per_candidate_acc.view(B, n)
            best_of_n_correct += (cand_matrix.sum(dim=-1) > 0).float().sum().item()
            n_problems += B

            batch_idx += 1
            print(f"[L2L Eval] batch {batch_idx} cum problems={n_problems} "
                  f"acc_no_exp={no_exp_correct/(n_problems*n):.4f} "
                  f"acc_with_exp_mean={with_exp_correct/n_candidates:.4f} "
                  f"acc_best_of_n={best_of_n_correct/n_problems:.4f}")

        if n_problems == 0:
            print("[L2L Eval] WARNING no val problems processed")
            return

        if phase_a_dump_only:
            # Copy local no_exp.jsonl to the canonical cache output path so
            # subsequent eval jobs can find it (--phase_a_cache <path>).
            if phase_a_cache_output and no_exp_path and os.path.exists(no_exp_path):
                import shutil
                os.makedirs(os.path.dirname(phase_a_cache_output), exist_ok=True)
                shutil.copy(no_exp_path, phase_a_cache_output)
                # fsync via re-open (shutil.copy doesn't fsync)
                with open(phase_a_cache_output, "rb") as _f:
                    os.fsync(_f.fileno())
                os.sync()
                print(f"[L2L Eval] Phase A cache written to {phase_a_cache_output} "
                      f"(n_problems={n_problems}, rollouts_per_problem={n}, "
                      f"acc_no_exp={no_exp_correct/(n_problems*n):.4f})")
            else:
                print(f"[L2L Eval] phase_a_dump_only: no cache output path set, "
                      f"local dump at {no_exp_path}")
            return

        # acc_no_exp samples = Phase A rollouts = n_problems * n
        acc_no_exp = no_exp_correct / (n_problems * n)
        acc_with_exp_mean = with_exp_correct / n_candidates
        acc_best_of_n = best_of_n_correct / n_problems
        metrics = {
            "eval/acc_no_exp": acc_no_exp,
            "eval/acc_with_exp_mean": acc_with_exp_mean,
            "eval/acc_best_of_n": acc_best_of_n,
            "eval/delta_with_exp_mean": acc_with_exp_mean - acc_no_exp,
            "eval/delta_best_of_n": acc_best_of_n - acc_no_exp,
            "eval/parse_success_rate": parse_success_count / max(n_candidates, 1),
            "eval/n_problems": n_problems,
            "eval/n_candidates": n_candidates,
            "eval/global_step": self.global_steps,
        }
        if compute_loss:
            def _finalize_loss(accumulator, suffix):
                if accumulator["count"] == 0:
                    return {}
                count = accumulator["count"]
                return {
                    f"eval/loss_{suffix}": accumulator["nll_mean"] / count,
                    f"eval/loss_{suffix}_sum": accumulator["nll_sum"] / count,
                    f"eval/response_len_{suffix}": accumulator["resp_len"] / count,
                }

            no_exp_metrics = _finalize_loss(loss_no_exp, "no_exp")
            with_exp_metrics = _finalize_loss(loss_with_exp, "with_exp")
            metrics.update(no_exp_metrics)
            metrics.update(with_exp_metrics)
            if no_exp_metrics and with_exp_metrics:
                metrics["eval/delta_loss"] = (
                    metrics["eval/loss_with_exp"] - metrics["eval/loss_no_exp"]
                )
                metrics["eval/delta_loss_sum"] = (
                    metrics["eval/loss_with_exp_sum"]
                    - metrics["eval/loss_no_exp_sum"]
                )
                metrics["eval/delta_response_len"] = (
                    metrics["eval/response_len_with_exp"]
                    - metrics["eval/response_len_no_exp"]
                )
        print(f"[L2L Eval] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        # Write summary.json so notebooks can read aggregated metrics directly
        if eval_dump_dir is not None:
            try:
                summary_path = os.path.join(eval_dump_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "experiment_name": self.config.trainer.experiment_name,
                        "global_step": self.global_steps,
                        "n": n,
                        "eval_max_problems": eval_max_problems,
                        **metrics,
                    }, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[L2L Eval] summary written to {summary_path}")
            except Exception as e:
                print(f"[L2L Eval Dump] summary.json write failed: {e}")

    def _run_l2l_eval_textgame(self, logger):
        """L2L textgame eval aligned with math _run_l2l_eval Phase A/B/C.

        For each unique env (B = held_out_size envs, seeded deterministically):
          1. Phase A — actor plays the env n times (one per rollout) with NO
             experience. n distinct sampled trajectories per env, each scored
             0/1 by the env's final reward.
          2. Phase B — exp_learner emits 1 experience per Phase A trajectory
             from (PREVIOUS="No previous experience.", LATEST=that trajectory's
             multi-round history), giving B*n candidate experiences.
          3. Phase C — actor re-plays each env once with the candidate
             experience prepended, using the SAME seed as Phase A so the env
             grid / start / goal are identical → 0/1 reward per candidate.

        Sokoban prologue stripping + Frozen Lake prologue stripping in the
        LATEST history mirrors _validate_textgame() exactly so the exp_learner
        sees the same format it was trained on.
        """
        EXPERIENCE_UPDATE_PROMPT = self.experience_update_prompt
        n = int(self.config.actor_rollout_ref.rollout.n)
        HELD_OUT_SIZE = int(self.config.trainer.get("held_out_size", 64))
        # Round down to actor world_size multiple (same constraint as
        # _validate_textgame): generate_sequences_textgame shards envs across
        # workers and asserts divisibility.
        HELD_OUT_SIZE = HELD_OUT_SIZE - HELD_OUT_SIZE % self.actor_rollout_wg.world_size
        assert HELD_OUT_SIZE > 0, f"held_out_size after rounding down to world_size multiple is 0"

        num_steps = int(self.config.trainer.textgame_max_steps)
        textgame_wfeedback = bool(self.config.trainer.textgame_wfeedback)

        # Deterministic seeds — same family as _validate_textgame() so this
        # eval is comparable across runs / ckpts.
        eval_seeds = [468382021 + 78025 + i * 1000 for i in range(HELD_OUT_SIZE)]

        phase_a_dump_only = bool(
            OmegaConf.select(self.config.trainer, "phase_a_dump_only", default=False)
        )
        phase_a_cache_output = OmegaConf.select(
            self.config.trainer, "phase_a_cache_output_path", default=None
        )
        phase_a_cache_path = OmegaConf.select(
            self.config.trainer, "phase_a_cache_path", default=None
        )

        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval_textgame")
        no_exp_path = os.path.join(eval_dump_dir, "no_exp.jsonl") if eval_dump_dir else None
        exp_gen_path = os.path.join(eval_dump_dir, "exp_gen.jsonl") if eval_dump_dir else None
        with_exp_path = os.path.join(eval_dump_dir, "with_exp_resolve.jsonl") if eval_dump_dir else None
        for p in (no_exp_path, exp_gen_path, with_exp_path):
            if p and os.path.exists(p):
                os.remove(p)

        # ── Phase A: actor plays each env n times with empty experience ────
        # generate_sequences_textgame consumes one (seed, experience) per env,
        # so to get n rollouts per env we loop n times with a seed offset that
        # only nudges the *sampling RNG* (vLLM seed inside the worker is
        # global; per-env seeds drive the env itself). For FrozenLake-v0-raw
        # the env is deterministic given seed, so identical seeds across the
        # n rounds keep the grid fixed while temperature produces distinct
        # trajectories.
        empty_exps = [""] * HELD_OUT_SIZE

        # rollout-major layout: rollout 0 first (all B envs), then rollout 1, ...
        # Flatten back to (B*n,) with index i = rollout_idx * B + env_idx so
        # downstream Phase B/C can index by `env_idx = i % B`.
        phase_a_trajs = [None] * (HELD_OUT_SIZE * n)
        phase_a_rewards = [0.0] * (HELD_OUT_SIZE * n)
        phase_a_final_feedback = [""] * (HELD_OUT_SIZE * n)

        if phase_a_cache_path:
            cache = {}
            with open(phase_a_cache_path, "r", encoding="utf-8") as cache_file:
                for line in cache_file:
                    record = json.loads(line)
                    cache[(int(record["env_idx"]), int(record["rollout_idx"]))] = record
            for rollout_idx in range(n):
                for env_idx in range(HELD_OUT_SIZE):
                    record = cache.get((env_idx, rollout_idx))
                    if record is None:
                        raise ValueError(
                            f"Phase A cache missing env={env_idx}, rollout={rollout_idx}; "
                            f"need {HELD_OUT_SIZE * n} records"
                        )
                    if int(record.get("seed", -1)) != eval_seeds[env_idx]:
                        raise ValueError(
                            f"Phase A cache seed mismatch at env={env_idx}: "
                            f"{record.get('seed')} != {eval_seeds[env_idx]}"
                        )
                    flat = rollout_idx * HELD_OUT_SIZE + env_idx
                    phase_a_trajs[flat] = {
                        "history": [
                            {
                                "step": step["step"],
                                "current_step_observation": step.get("observation", ""),
                                "raw_response": step.get("response", ""),
                            }
                            for step in record.get("history", [])
                        ],
                        "stop_reason": record.get("stop_reason", "unknown"),
                        "final_feedback": record.get("final_feedback", ""),
                    }
                    phase_a_rewards[flat] = float(record["reward"])
                    phase_a_final_feedback[flat] = record.get("final_feedback", "")
        else:
            for rollout_idx in range(n):
                out = self.actor_rollout_wg.generate_sequences_textgame(
                    env_config=self.textgame_env_config,
                    env_num=HELD_OUT_SIZE,
                    tokenizer=self.tokenizer,
                    experiences=empty_exps,
                    num_steps=num_steps,
                    seeds=eval_seeds,
                    validate=True,
                )
                if isinstance(out, list):
                    out = out[0]
                trajs = out["env_trajectories"]
                rewards = [rd[0] for rd in out["reward_list"]]
                for env_idx in range(HELD_OUT_SIZE):
                    flat = rollout_idx * HELD_OUT_SIZE + env_idx
                    phase_a_trajs[flat] = trajs.get(env_idx, {})
                    phase_a_rewards[flat] = float(rewards[env_idx])
                    phase_a_final_feedback[flat] = phase_a_trajs[flat].get(
                        "final_feedback", ""
                    )

        phase_a_acc = torch.tensor(phase_a_rewards, dtype=torch.float32)
        no_exp_correct = (phase_a_acc == 1.0).float().sum().item()

        # Phase A dump
        if no_exp_path is not None:
            try:
                recs = []
                for i in range(HELD_OUT_SIZE * n):
                    env_idx = i % HELD_OUT_SIZE
                    rollout_idx = i // HELD_OUT_SIZE
                    traj = phase_a_trajs[i]
                    history = traj.get("history", [])
                    recs.append({
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "reward": phase_a_rewards[i],
                        "correct": bool(phase_a_rewards[i] == 1.0),
                        "n_steps": len(history),
                        "stop_reason": traj.get("stop_reason", "unknown"),
                        "history": [
                            {"step": s["step"],
                             "observation": s.get("current_step_observation", ""),
                             "response": s.get("raw_response", "")}
                            for s in history
                        ],
                        "final_feedback": phase_a_final_feedback[i],
                    })
                self._dump_jsonl(no_exp_path, recs, append=False)
            except Exception as e:
                print(f"[L2L Eval Textgame Dump] no_exp dump failed: {e}")

        if phase_a_dump_only:
            if phase_a_cache_output and no_exp_path and os.path.exists(no_exp_path):
                import shutil

                os.makedirs(os.path.dirname(phase_a_cache_output), exist_ok=True)
                shutil.copy(no_exp_path, phase_a_cache_output)
                with open(phase_a_cache_output, "rb") as cache_file:
                    os.fsync(cache_file.fileno())
                os.sync()
                print(
                    f"[L2L Eval Textgame] Phase A cache written to "
                    f"{phase_a_cache_output}"
                )
            logger.log(
                data={
                    "eval/acc_no_exp": no_exp_correct / max(HELD_OUT_SIZE * n, 1),
                    "eval/n_problems": HELD_OUT_SIZE,
                    "eval/n_candidates": HELD_OUT_SIZE * n,
                    "eval/global_step": self.global_steps,
                },
                step=self.global_steps,
            )
            return

        # ── Phase B: exp_learner emits 1 experience per Phase A trajectory ──
        _strip_env_prologue = self._l2c_strip_env_prologue

        exp_learner_prompts = []
        for i in range(HELD_OUT_SIZE * n):
            traj = phase_a_trajs[i]
            history = traj.get("history", [])
            multi_full_history = ""
            for step_info in history:
                step_num = step_info["step"]
                obs = _strip_env_prologue(step_info.get("current_step_observation", ""), step_num)
                raw_response = step_info.get("raw_response", "")
                multi_full_history += f"\nRound{step_num}_Input: {obs}\n\nRound{step_num}_Output: {raw_response}"
            if textgame_wfeedback:
                multi_full_history += f"\n\n\n{traj.get('final_feedback', '')}\n"
            exp_learner_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                PREVIOUS_EXPERIENCE="No previous experience.",
                LATEST_EXPERIENCE=multi_full_history,
            ))

        exp_learner_batch = self._l2c_tokenize_textgame_prompts(
            exp_learner_prompts
        )
        exp_learner_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
            "n": 1,
        }
        exp_padded, exp_pad_size = pad_dataproto_to_divisor(
            exp_learner_batch, self.exp_learner_wg.world_size
        )
        exp_out_padded = self.exp_learner_wg.generate_sequences(exp_padded)
        exp_out = unpad_dataproto(exp_out_padded, exp_pad_size)

        experiences = []
        raw_exp_texts = []
        parsed_exps = []
        parse_oks = []
        parse_success_count = 0
        max_exp_tokens = self.config.trainer.experience_max_length
        for i in range(HELD_OUT_SIZE * n):
            exp_text = self.tokenizer.decode(exp_out.batch["responses"][i], skip_special_tokens=True)
            raw_exp_texts.append(exp_text)
            parsed = self._parse_experience(exp_text)
            parsed_exps.append(parsed)
            ok = bool(parsed)
            parse_oks.append(ok)
            if ok:
                parse_success_count += 1
            # If parsing fails, fall back to empty experience for Phase C so
            # the candidate effectively measures "no experience" — same
            # semantics as math _run_l2l_eval's combined="No previous experience."
            # fallback (the textgame builder treats "" as "skip experience block").
            combined = self._truncate_experience(parsed, max_exp_tokens) if parsed else ""
            experiences.append(combined)

        # Phase B dump
        if exp_gen_path is not None:
            try:
                recs = []
                for i in range(HELD_OUT_SIZE * n):
                    env_idx = i % HELD_OUT_SIZE
                    rollout_idx = i // HELD_OUT_SIZE
                    recs.append({
                        "candidate_idx": i,
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "exp_learner_input": exp_learner_prompts[i],
                        "raw_output": raw_exp_texts[i],
                        "parsed_exp": parsed_exps[i],
                        "parse_ok": parse_oks[i],
                        "experience_used": experiences[i],
                    })
                self._dump_jsonl(exp_gen_path, recs, append=False)
            except Exception as e:
                print(f"[L2L Eval Textgame Dump] exp_gen dump failed: {e}")

        # ── Phase C: actor re-plays each env with the candidate experience ──
        # Same seed as Phase A so env state (grid, start, goal) is identical.
        # We run n rounds, each round pairs (env_idx) with (its rollout_idx's
        # candidate). i = rollout_idx * HELD_OUT_SIZE + env_idx.
        phase_c_rewards = [0.0] * (HELD_OUT_SIZE * n)
        phase_c_trajs = [None] * (HELD_OUT_SIZE * n)
        for rollout_idx in range(n):
            round_exps = experiences[rollout_idx * HELD_OUT_SIZE:(rollout_idx + 1) * HELD_OUT_SIZE]
            out = self.actor_rollout_wg.generate_sequences_textgame(
                env_config=self.textgame_env_config,
                env_num=HELD_OUT_SIZE,
                tokenizer=self.tokenizer,
                experiences=round_exps,
                num_steps=num_steps,
                seeds=eval_seeds,
                validate=True,
            )
            if isinstance(out, list):
                out = out[0]
            trajs = out["env_trajectories"]
            rewards = [rd[0] for rd in out["reward_list"]]
            for env_idx in range(HELD_OUT_SIZE):
                flat = rollout_idx * HELD_OUT_SIZE + env_idx
                phase_c_trajs[flat] = trajs.get(env_idx, {})
                phase_c_rewards[flat] = float(rewards[env_idx])

        per_candidate_acc = (torch.tensor(phase_c_rewards, dtype=torch.float32) == 1.0).float()
        with_exp_correct = per_candidate_acc.sum().item()

        # Phase C dump
        if with_exp_path is not None:
            try:
                recs = []
                for i in range(HELD_OUT_SIZE * n):
                    env_idx = i % HELD_OUT_SIZE
                    rollout_idx = i // HELD_OUT_SIZE
                    traj = phase_c_trajs[i]
                    history = traj.get("history", [])
                    recs.append({
                        "candidate_idx": i,
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "experience_used": experiences[i],
                        "reward": phase_c_rewards[i],
                        "correct": bool(phase_c_rewards[i] == 1.0),
                        "n_steps": len(history),
                        "stop_reason": traj.get("stop_reason", "unknown"),
                        "history": [
                            {"step": s["step"],
                             "observation": s.get("current_step_observation", ""),
                             "response": s.get("raw_response", "")}
                            for s in history
                        ],
                    })
                self._dump_jsonl(with_exp_path, recs, append=False)
            except Exception as e:
                print(f"[L2L Eval Textgame Dump] with_exp dump failed: {e}")

        # ── Aggregate metrics ──────────────────────────────────────────────
        # Per-env layout: candidates for env e are at flat indices
        # rollout_idx * B + e for rollout_idx in [0, n). Reshape to (n, B)
        # then transpose so dim 0 = env, dim 1 = candidate.
        B = HELD_OUT_SIZE
        cand_matrix = per_candidate_acc.view(n, B).t().contiguous()  # (B, n)
        best_of_n_correct = (cand_matrix.sum(dim=-1) > 0).float().sum().item()
        n_candidates = B * n
        n_problems = B

        acc_no_exp = no_exp_correct / max(n_candidates, 1)
        acc_with_exp_mean = with_exp_correct / max(n_candidates, 1)
        acc_best_of_n = best_of_n_correct / max(n_problems, 1)
        metrics = {
            "eval/acc_no_exp": acc_no_exp,
            "eval/acc_with_exp_mean": acc_with_exp_mean,
            "eval/acc_best_of_n": acc_best_of_n,
            "eval/delta_with_exp_mean": acc_with_exp_mean - acc_no_exp,
            "eval/delta_best_of_n": acc_best_of_n - acc_no_exp,
            "eval/parse_success_rate": parse_success_count / max(n_candidates, 1),
            "eval/n_problems": n_problems,
            "eval/n_candidates": n_candidates,
            "eval/global_step": self.global_steps,
        }
        print(f"[L2L Eval Textgame] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if eval_dump_dir is not None:
            try:
                summary_path = os.path.join(eval_dump_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "experiment_name": self.config.trainer.experiment_name,
                        "global_step": self.global_steps,
                        "n": n,
                        "held_out_size": HELD_OUT_SIZE,
                        "textgame_env_id": self.config.trainer.textgame_env_id,
                        **metrics,
                    }, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[L2L Eval Textgame] summary written to {summary_path}")
            except Exception as e:
                print(f"[L2L Eval Textgame Dump] summary.json write failed: {e}")

    def _run_l2l_eval_iterative_textgame(self, logger, iter_compact_steps):
        """Iterative-compact textgame eval: test-time scaling via repeated
        exp_learner compaction on the SAME envs (fixed seeds per env). Mirror
        of _run_l2l_eval_iterative (math) but adapted to textgame env stepping.

        For each of B = held_out_size envs, n rollouts per env:
          iter 0:  actor plays env with empty exp  →  traj_0, reward_0
          for k in 1..K-1:
              exp_learner((multi-round history from traj_{k-1}),
                           PREV = carry_prev ? exp_{k-1} : "No previous exp.")
                  → exp_k
              actor RE-PLAYS same env (SAME seed) with exp_k prepended
                  → traj_k, reward_k

        The seed is fixed per env across all K iters, so grid/goal are
        identical; only the experience (and thus the actor's plan) varies.
        This is textgame's analog of "same problem, refine and re-solve".

        Logs:
          eval/iter_{k}/acc        for k = 0..K-1  (per-candidate mean)
          eval/iter_{k}/parse_ok   for k = 1..K-1  (exp_learner parse rate)
          eval/best_iter_acc       per-candidate OR across iters
        plus per-iter dumps no_exp.jsonl (k=0), exp_gen_iter_{k}.jsonl,
        with_exp_resolve_iter_{k}.jsonl, and summary.json.

        Structural differences vs math _run_l2l_eval_iterative:
          - No val_dataloader — envs are enumerated by seed once, no batch
            iteration loop.
          - Tokenization goes through apply_chat_template (textgame has no
            train_dataset.re_tokenize), matching _run_l2l_eval_textgame.
          - Phase A / C use generate_sequences_textgame (env stepping), not
            actor.generate on a raw prompt.
        """
        EXPERIENCE_UPDATE_PROMPT = self.experience_update_prompt
        n = int(self.config.actor_rollout_ref.rollout.n)
        K = int(iter_compact_steps)
        assert K >= 2, f"_run_l2l_eval_iterative_textgame requires K>=2, got {K}"
        carry_prev = bool(OmegaConf.select(self.config.trainer, "iter_compact_carry_prev", default=True))

        HELD_OUT_SIZE = int(self.config.trainer.get("held_out_size", 64))
        # Round down to actor world_size multiple (same constraint as
        # _validate_textgame and _run_l2l_eval_textgame).
        HELD_OUT_SIZE = HELD_OUT_SIZE - HELD_OUT_SIZE % self.actor_rollout_wg.world_size
        assert HELD_OUT_SIZE > 0, "held_out_size after rounding to world_size multiple is 0"

        num_steps = int(self.config.trainer.textgame_max_steps)

        # Deterministic seeds — same family as _validate_textgame() /
        # _run_l2l_eval_textgame() so metrics are comparable across runs.
        eval_seeds = [468382021 + 78025 + i * 1000 for i in range(HELD_OUT_SIZE)]

        # Dump paths (per-iter files, iter 0 named no_exp for join-compat with
        # notebooks that key off k=0 -> no_exp).
        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval_iter_textgame")
        def _path(name):
            return os.path.join(eval_dump_dir, name) if eval_dump_dir else None
        no_exp_path = _path("no_exp.jsonl")
        exp_gen_paths = [_path(f"exp_gen_iter_{k}.jsonl") for k in range(K)]
        resolve_paths = [_path(f"with_exp_resolve_iter_{k}.jsonl") for k in range(K)]
        for p in [no_exp_path] + exp_gen_paths + resolve_paths:
            if p and os.path.exists(p):
                os.remove(p)

        max_exp_tokens = self.config.trainer.experience_max_length

        def _traj_to_history_text(traj):
            return self._l2c_textgame_history(traj)

        # ── Per-iter aggregates ──────────────────────────────────────────────
        correct_per_iter = [0.0] * K
        parse_ok_per_iter = [0] * K   # k>=1 only, k=0 stays 0

        # ── iter 0: Phase A (empty experience) ───────────────────────────────
        # Rollout-major layout: flat = rollout_idx * B + env_idx, matching
        # _run_l2l_eval_textgame.
        B = HELD_OUT_SIZE
        empty_exps = [""] * B
        cur_trajs = [None] * (B * n)
        iter0_rewards = [0.0] * (B * n)
        for rollout_idx in range(n):
            out = self.actor_rollout_wg.generate_sequences_textgame(
                env_config=self.textgame_env_config,
                env_num=B,
                tokenizer=self.tokenizer,
                experiences=empty_exps,
                num_steps=num_steps,
                seeds=eval_seeds,
                validate=True,
            )
            if isinstance(out, list):
                out = out[0]
            trajs = out["env_trajectories"]
            rewards = [rd[0] for rd in out["reward_list"]]
            for env_idx in range(B):
                flat = rollout_idx * B + env_idx
                cur_trajs[flat] = trajs.get(env_idx, {})
                iter0_rewards[flat] = float(rewards[env_idx])

        iter0_acc = (torch.tensor(iter0_rewards, dtype=torch.float32) == 1.0).float()
        correct_per_iter[0] += iter0_acc.sum().item()
        per_cand_best = iter0_acc.clone()

        # iter 0 dump — matches _run_l2l_eval_textgame's no_exp.jsonl schema
        # so iter=0 output is byte-for-byte comparable to single-K eval.
        if no_exp_path is not None:
            try:
                recs = []
                for i in range(B * n):
                    env_idx = i % B
                    rollout_idx = i // B
                    traj = cur_trajs[i]
                    history = traj.get("history", [])
                    recs.append({
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "iter_k": 0,
                        "reward": iter0_rewards[i],
                        "correct": bool(iter0_rewards[i] == 1.0),
                        "n_steps": len(history),
                        "stop_reason": traj.get("stop_reason", "unknown"),
                        "history": [
                            {"step": s["step"],
                             "observation": s.get("current_step_observation", ""),
                             "response": s.get("raw_response", "")}
                            for s in history
                        ],
                    })
                self._dump_jsonl(no_exp_path, recs, append=False)
            except Exception as e:
                print(f"[L2L Eval Iter Textgame Dump] iter 0 dump failed: {e}")

        # ── Per-candidate compact state; C_0 = "" (no compact yet) ───────────
        experiences = [""] * (B * n)
        prev_experiences = [""] * (B * n)

        # ── iter 1..K-1 ──────────────────────────────────────────────────────
        for k in range(1, K):
            # Phase B: exp_learner((history_{k-1}), PREV=carry ? exp_{k-1} : "No previous exp")
            exp_learner_prompts = []
            latest_texts = []
            for i in range(B * n):
                latest = _traj_to_history_text(cur_trajs[i])
                latest_texts.append(latest)
                prev = prev_experiences[i] if (carry_prev and prev_experiences[i]) else "No previous experience."
                exp_learner_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                    PREVIOUS_EXPERIENCE=prev,
                    LATEST_EXPERIENCE=latest,
                ))

            exp_learner_batch = self._l2c_tokenize_textgame_prompts(
                exp_learner_prompts
            )
            exp_learner_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "n": 1,
            }
            exp_padded, exp_pad_size = pad_dataproto_to_divisor(
                exp_learner_batch, self.exp_learner_wg.world_size
            )
            exp_out_padded = self.exp_learner_wg.generate_sequences(exp_padded)
            exp_out = unpad_dataproto(exp_out_padded, exp_pad_size)

            raw_exp_texts, parsed_exps, parse_oks, new_experiences = [], [], [], []
            for i in range(B * n):
                exp_text = self.tokenizer.decode(exp_out.batch["responses"][i], skip_special_tokens=True)
                raw_exp_texts.append(exp_text)
                parsed = self._parse_experience(exp_text)
                parsed_exps.append(parsed)
                ok = bool(parsed)
                parse_oks.append(ok)
                if ok:
                    parse_ok_per_iter[k] += 1
                # Fall back to prev exp on parse fail — same semantic as math
                # iter eval. At k=1 with no prev, this falls back to "" (empty
                # experience block, treated as "skip" by the textgame builder).
                combined = parsed if parsed else prev_experiences[i]
                combined = self._truncate_experience(combined, max_exp_tokens) if combined else ""
                new_experiences.append(combined)
            experiences = new_experiences

            # Phase B dump
            if exp_gen_paths[k] is not None:
                try:
                    recs = []
                    for i in range(B * n):
                        env_idx = i % B
                        rollout_idx = i // B
                        recs.append({
                            "candidate_idx": i,
                            "env_idx": env_idx,
                            "rollout_idx": rollout_idx,
                            "seed": eval_seeds[env_idx],
                            "iter_k": k,
                            "prev_experience": prev_experiences[i],
                            "latest_history": latest_texts[i],
                            "exp_learner_input": exp_learner_prompts[i],
                            "raw_output": raw_exp_texts[i],
                            "parsed_exp": parsed_exps[i],
                            "parse_ok": parse_oks[i],
                            "experience_used": experiences[i],
                        })
                    self._dump_jsonl(exp_gen_paths[k], recs, append=False)
                except Exception as e:
                    print(f"[L2L Eval Iter Textgame Dump] exp_gen iter {k} dump failed: {e}")

            # Phase C: actor RE-PLAYS each env with candidate exp; SAME seeds.
            new_trajs = [None] * (B * n)
            new_rewards = [0.0] * (B * n)
            for rollout_idx in range(n):
                round_exps = experiences[rollout_idx * B:(rollout_idx + 1) * B]
                out = self.actor_rollout_wg.generate_sequences_textgame(
                    env_config=self.textgame_env_config,
                    env_num=B,
                    tokenizer=self.tokenizer,
                    experiences=round_exps,
                    num_steps=num_steps,
                    seeds=eval_seeds,
                    validate=True,
                )
                if isinstance(out, list):
                    out = out[0]
                trajs = out["env_trajectories"]
                rewards = [rd[0] for rd in out["reward_list"]]
                for env_idx in range(B):
                    flat = rollout_idx * B + env_idx
                    new_trajs[flat] = trajs.get(env_idx, {})
                    new_rewards[flat] = float(rewards[env_idx])

            iter_k_acc = (torch.tensor(new_rewards, dtype=torch.float32) == 1.0).float()
            correct_per_iter[k] += iter_k_acc.sum().item()
            per_cand_best = torch.max(per_cand_best, iter_k_acc)

            # Phase C dump
            if resolve_paths[k] is not None:
                try:
                    recs = []
                    for i in range(B * n):
                        env_idx = i % B
                        rollout_idx = i // B
                        traj = new_trajs[i]
                        history = traj.get("history", [])
                        recs.append({
                            "candidate_idx": i,
                            "env_idx": env_idx,
                            "rollout_idx": rollout_idx,
                            "seed": eval_seeds[env_idx],
                            "iter_k": k,
                            "experience_used": experiences[i],
                            "reward": new_rewards[i],
                            "correct": bool(new_rewards[i] == 1.0),
                            "n_steps": len(history),
                            "stop_reason": traj.get("stop_reason", "unknown"),
                            "history": [
                                {"step": s["step"],
                                 "observation": s.get("current_step_observation", ""),
                                 "response": s.get("raw_response", "")}
                                for s in history
                            ],
                        })
                    self._dump_jsonl(resolve_paths[k], recs, append=False)
                except Exception as e:
                    print(f"[L2L Eval Iter Textgame Dump] resolve iter {k} dump failed: {e}")

            # Roll state forward for next iter.
            cur_trajs = new_trajs
            prev_experiences = experiences

        # ── Aggregate ────────────────────────────────────────────────────────
        n_candidates = B * n
        n_problems = B
        best_iter_correct = per_cand_best.sum().item()

        metrics = {f"eval/iter_{k}/acc": correct_per_iter[k] / max(n_candidates, 1) for k in range(K)}
        for k in range(1, K):
            metrics[f"eval/iter_{k}/parse_ok"] = parse_ok_per_iter[k] / max(n_candidates, 1)
        metrics["eval/best_iter_acc"] = best_iter_correct / max(n_candidates, 1)
        metrics["eval/n_problems"] = n_problems
        metrics["eval/n_candidates"] = n_candidates
        metrics["eval/global_step"] = self.global_steps
        metrics["eval/iter_compact_steps"] = K
        metrics["eval/iter_compact_carry_prev"] = int(carry_prev)
        print(f"[L2L Eval Iter Textgame] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if eval_dump_dir is not None:
            try:
                summary_path = os.path.join(eval_dump_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "experiment_name": self.config.trainer.experiment_name,
                        "global_step": self.global_steps,
                        "n": n,
                        "K": K,
                        "carry_prev": carry_prev,
                        "held_out_size": B,
                        "textgame_env_id": self.config.trainer.textgame_env_id,
                        **{k: (v if not isinstance(v, torch.Tensor) else v.item()) for k, v in metrics.items()},
                    }, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[L2L Eval Iter Textgame] summary written to {summary_path}")
            except Exception as e:
                print(f"[L2L Eval Iter Textgame Dump] summary.json write failed: {e}")

    def _run_l2l_eval_iterative(self, logger, iter_compact_steps):
        """Iterative-compact eval: test-time scaling via repeated exp_learner
        compaction. For each val problem:
          iter 0: actor solves with NO experience  →  A_0, acc_at_iter_0
          for k in 1..K-1:
              exp_learner((Q, A_{k-1}), PREV=C_{k-1}) → C_k
              actor solves with C_k prepended       → A_k, acc_at_iter_k

        C_0 = "" (no prior compact). PREV semantics chosen by
        ``iter_compact_carry_prev`` (default True): pass C_{k-1} as
        PREVIOUS_EXPERIENCE in the EXPERIENCE_UPDATE_PROMPT slot. False mode
        always uses "No previous experience." (in-distribution for vanilla
        ckpts trained with exp_sel_with_prev=False).

        At iter k>0, the actor's prompt is EXPERIENCE_SOLVE_PROMPT_TEMPLATE
        wrapped with C_k — same path as Phase C in _run_l2l_eval. So
        _compute_binary_reward_math is reusable: just feed it the per-source
        experience list of length B*n (here n=rollout.n, by default 1 per the
        recommended iterative-eval setup).

        Logs:
          eval/iter_{k}/acc            for k = 0..K-1
          eval/iter_{k}/parse_ok       (parse success rate of exp_learner at k)
          eval/best_iter_acc           per-problem max acc across iters
        plus per-iter dumps no_exp.jsonl (k=0) / exp_gen_iter_{k}.jsonl /
        with_exp_resolve_iter_{k}.jsonl, and summary.json with per-iter
        metrics aggregated across all batches.

        Why a separate function vs threading a loop through _run_l2l_eval:
        keeps the single-K path byte-identical (so cached Phase A semantics
        don't drift) and lets the K-loop reason about per-iter state without
        the cache / phase_a_dump_only branches.
        """
        EXPERIENCE_UPDATE_PROMPT = self.experience_update_prompt
        n = self.config.actor_rollout_ref.rollout.n
        K = int(iter_compact_steps)
        assert K >= 2, f"_run_l2l_eval_iterative requires K>=2, got {K}"
        eval_max_problems = int(self.config.trainer.get("eval_max_problems", 1000))
        carry_prev = bool(OmegaConf.select(self.config.trainer, "iter_compact_carry_prev", default=True))

        # Per-iter dump files; iter 0 reuses no_exp filename for join-compat
        # with notebooks that already key off (k=0 -> no_exp).
        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval_iter")
        def _path(name):
            return os.path.join(eval_dump_dir, name) if eval_dump_dir else None
        no_exp_path = _path("no_exp.jsonl")
        exp_gen_paths = [_path(f"exp_gen_iter_{k}.jsonl") for k in range(K)]
        resolve_paths = [_path(f"with_exp_resolve_iter_{k}.jsonl") for k in range(K)]
        for p in [no_exp_path] + exp_gen_paths + resolve_paths:
            if p and os.path.exists(p):
                os.remove(p)

        # Per-iter aggregates
        correct_per_iter = [0.0] * K      # numerator: sum of per-candidate 0/1
        parse_ok_per_iter = [0] * K       # exp_learner parse counts (k>=1 only)
        n_candidates = 0                  # B*n total candidates (denominator)
        n_problems = 0                    # B total source problems

        # best_of_iters: per-problem OR across all K iters. Aggregated per batch
        # then summed.
        best_iter_correct = 0.0

        # Optional Phase A cache for iter 0 — same file format as the single-K
        # eval cache (one JSONL row per (global_problem_idx, rollout_idx) with
        # 'response_text' + 'correct' + 'full_prompt'). Sharing the cache with
        # _run_l2l_eval has two upsides: (1) cross-eval consistency — iter 0
        # acc here matches acc_no_exp in single-K eval byte-for-byte, and (2)
        # eliminates iter-0 vLLM sampling noise so cross-ckpt comparisons at
        # iter k>=1 attribute differences entirely to the exp_learner. The
        # cache is keyed by problem index only (actor is the frozen base, no
        # ckpt dependency), so it's reusable across all eval runs on a given
        # model size. iter k>=1 still calls actor.generate live (the prompt
        # changes per iter).
        _phase_a_cache_path = OmegaConf.select(self.config.trainer, "phase_a_cache_path", default=None)
        _phase_a_cache: dict = {}
        if _phase_a_cache_path:
            with open(_phase_a_cache_path, "r", encoding="utf-8") as _cf:
                for _line in _cf:
                    _r = json.loads(_line)
                    _phase_a_cache.setdefault(_r["global_problem_idx"], []).append(_r)
            for _gpi in _phase_a_cache:
                _phase_a_cache[_gpi].sort(key=lambda x: x["rollout_idx"])
            print(f"[L2L Eval Iter] Phase A cache loaded for iter 0: "
                  f"{len(_phase_a_cache)} problems from {_phase_a_cache_path}")

        val_iter = iter(self.val_dataloader)
        batch_idx = 0

        while n_problems < eval_max_problems:
            try:
                batch_dict = next(val_iter)
            except StopIteration:
                print(f"[L2L Eval Iter] val dataloader exhausted at {n_problems}/{eval_max_problems}")
                break

            batch = DataProto.from_single_dict(batch_dict)
            remaining = eval_max_problems - n_problems
            if len(batch) > remaining:
                batch = batch[:remaining]
            B = len(batch)
            source_raw_prompts = list(batch.non_tensor_batch['raw_prompt'])

            # ── iter 0: Phase A (no experience) ──────────────────────────────
            if _phase_a_cache:
                # Cache hit: pull iter_responses + correctness from disk; skip
                # actor.generate entirely. Same key scheme as _run_l2l_eval —
                # each src problem gets the first n cached rollouts in order.
                _cache_recs = []
                for _bi in range(B):
                    _gpi = n_problems + _bi
                    _recs = _phase_a_cache.get(_gpi, [])
                    if len(_recs) < n:
                        raise ValueError(
                            f"Phase A cache missing rollouts for global_problem_idx={_gpi}: "
                            f"need {n}, have {len(_recs)}"
                        )
                    _cache_recs.extend(_recs[:n])
                iter_responses = [_r["response_text"] for _r in _cache_recs]
                iter0_acc = torch.tensor([float(_r["correct"]) for _r in _cache_recs])
                correct_per_iter[0] += iter0_acc.sum().item()
                per_cand_best = iter0_acc.clone()
                if no_exp_path is not None:
                    try:
                        recs = []
                        for i, _r in enumerate(_cache_recs):
                            rec = dict(_r)
                            rec["batch_idx"] = batch_idx
                            rec["problem_idx_in_batch"] = i // n
                            rec["rollout_idx"] = i % n
                            rec["global_problem_idx"] = n_problems + (i // n)
                            recs.append(rec)
                        self._dump_jsonl(no_exp_path, recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Iter Dump] iter 0 cache copy failed: {e}")
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
                )
                gen_batch.non_tensor_batch.pop("raw_prompt_ids", None)
                gen_batch.non_tensor_batch.pop("raw_prompt", None)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": True,
                }
                gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                gen_padded.meta_info["n"] = n
                gen_out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                gen_out = unpad_dataproto(gen_out_padded, pad_size * n)

                src_indices = np.array([i // n for i in range(B * n)])
                for key in ("reward_model", "data_source"):
                    if key in batch.non_tensor_batch:
                        gen_out.non_tensor_batch[key] = batch.non_tensor_batch[key][src_indices]
                iter0_reward, _ = compute_reward(gen_out, self.reward_fn)
                iter0_acc = (iter0_reward.sum(-1) == 1.0).float()    # (B*n,)
                correct_per_iter[0] += iter0_acc.sum().item()

                iter_responses = [
                    self.tokenizer.decode(gen_out.batch["responses"][i], skip_special_tokens=True)
                    for i in range(B * n)
                ]

                # Per-problem best-acc tracker (across iters); start with iter 0
                per_cand_best = iter0_acc.clone()

                # iter 0 dump (matches no_exp.jsonl schema in _run_l2l_eval)
                if no_exp_path is not None:
                    try:
                        rm = batch.non_tensor_batch.get("reward_model", None)
                        ds = batch.non_tensor_batch.get("data_source", None)
                        acc_list = iter0_acc.cpu().tolist()
                        full_prompts = self._decode_prompts_batch(gen_out.batch["prompts"])
                        recs = []
                        for i in range(B * n):
                            src_idx = i // n
                            recs.append({
                                "batch_idx": batch_idx,
                                "problem_idx_in_batch": src_idx,
                                "rollout_idx": i % n,
                                "global_problem_idx": n_problems + src_idx,
                                "prompt_text": source_raw_prompts[src_idx][-1]["content"],
                                "full_prompt": full_prompts[i],
                                "response_text": iter_responses[i],
                                "ground_truth": rm[src_idx].get("ground_truth") if rm is not None else None,
                                "data_source": str(ds[src_idx]) if ds is not None else None,
                                "correct": bool(acc_list[i] == 1.0),
                            })
                        self._dump_jsonl(no_exp_path, recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Iter Dump] iter 0 dump failed: {e}")

            # Per-candidate compact state, length B*n. C_0 = "" (no compact yet).
            experiences = [""] * (B * n)
            prev_experiences = [""] * (B * n)   # PREV passed into exp_learner at iter k

            # ── iter 1..K-1 ───────────────────────────────────────────────────
            for k in range(1, K):
                # Phase B at iter k: exp_learner((Q, A_{k-1}), PREV=carry?prev:None)
                exp_prompts = []
                for i in range(B * n):
                    src_idx = i // n
                    latest = f"Input: {source_raw_prompts[src_idx][-1]['content']}\nOutput: {iter_responses[i]}"
                    prev = prev_experiences[i] if (carry_prev and prev_experiences[i]) else "No previous experience."
                    exp_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                        PREVIOUS_EXPERIENCE=prev,
                        LATEST_EXPERIENCE=latest,
                    ))

                exp_tokenized = [self.train_dataset.re_tokenize([{"role": "user", "content": p}])
                                 for p in exp_prompts]
                exp_learner_batch = DataProto.from_single_dict({
                    "input_ids": torch.stack([t["input_ids"] for t in exp_tokenized]),
                    "attention_mask": torch.stack([t["attention_mask"] for t in exp_tokenized]),
                    "position_ids": torch.stack([t["position_ids"] for t in exp_tokenized]),
                })
                exp_learner_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": True,
                    "n": 1,
                }
                exp_padded, exp_pad_size = pad_dataproto_to_divisor(exp_learner_batch, self.exp_learner_wg.world_size)
                exp_out_padded = self.exp_learner_wg.generate_sequences(exp_padded)
                exp_out = unpad_dataproto(exp_out_padded, exp_pad_size)

                raw_exp_texts = []
                parsed_exps = []
                parse_oks = []
                new_experiences = []
                for i in range(B * n):
                    exp_text = self.tokenizer.decode(exp_out.batch["responses"][i], skip_special_tokens=True)
                    raw_exp_texts.append(exp_text)
                    parsed = self._parse_experience(exp_text)
                    parsed_exps.append(parsed)
                    ok = bool(parsed)
                    parse_oks.append(ok)
                    if ok:
                        parse_ok_per_iter[k] += 1
                    # Use parsed if non-empty; otherwise fall back to previous
                    # experience (don't reset to empty — that would lose
                    # accumulated compact state across iters).
                    combined = parsed if parsed else prev_experiences[i]
                    combined = self._truncate_experience(combined, self.config.trainer.experience_max_length)
                    new_experiences.append(combined)
                experiences = new_experiences

                # Phase B dump at iter k
                if exp_gen_paths[k] is not None:
                    try:
                        full_prompts = self._decode_prompts_batch(exp_out.batch["prompts"])
                        recs = []
                        for i in range(B * n):
                            src_idx = i // n
                            recs.append({
                                "batch_idx": batch_idx,
                                "candidate_idx_in_batch": i,
                                "source_idx_in_batch": src_idx,
                                "rollout_idx": i % n,
                                "global_source_idx": n_problems + src_idx,
                                "iter_k": k,
                                "prev_experience": prev_experiences[i],
                                "exp_learner_input": exp_prompts[i],
                                "full_prompt": full_prompts[i],
                                "raw_output": raw_exp_texts[i],
                                "parsed_exp": parsed_exps[i],
                                "parse_ok": parse_oks[i],
                                "combined_truncated": experiences[i],
                            })
                        self._dump_jsonl(exp_gen_paths[k], recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Iter Dump] exp_gen iter {k} dump failed: {e}")

                # Phase C at iter k: actor re-solves with C_k prepended.
                per_cand_acc, resolve_responses_text, resolve_full_prompts, _ = self._compute_binary_reward_math(
                    experiences, batch, source_raw_prompts, n,
                    return_dump_info=True,
                )  # (B*n,) in {0, 1}

                correct_per_iter[k] += per_cand_acc.sum().item()
                per_cand_best = torch.max(per_cand_best, per_cand_acc)

                # Phase C dump at iter k
                if resolve_paths[k] is not None:
                    try:
                        rm = batch.non_tensor_batch.get("reward_model", None)
                        ds = batch.non_tensor_batch.get("data_source", None)
                        rewards_list = per_cand_acc.cpu().tolist()
                        recs = []
                        for i in range(B * n):
                            src_idx = i // n
                            recs.append({
                                "batch_idx": batch_idx,
                                "candidate_idx_in_batch": i,
                                "source_idx_in_batch": src_idx,
                                "rollout_idx": i % n,
                                "global_source_idx": n_problems + src_idx,
                                "iter_k": k,
                                "experience_used": experiences[i],
                                "problem_text": source_raw_prompts[src_idx][-1]["content"],
                                "full_prompt": resolve_full_prompts[i],
                                "response_text": resolve_responses_text[i],
                                "reward": float(rewards_list[i]),
                                "ground_truth": rm[src_idx].get("ground_truth") if rm is not None else None,
                                "data_source": str(ds[src_idx]) if ds is not None else None,
                            })
                        self._dump_jsonl(resolve_paths[k], recs, append=True)
                    except Exception as e:
                        print(f"[L2L Eval Iter Dump] resolve iter {k} dump failed: {e}")

                # Roll forward state for next iter.
                iter_responses = resolve_responses_text
                prev_experiences = experiences

            n_candidates += B * n
            n_problems += B
            best_iter_correct += per_cand_best.sum().item()
            batch_idx += 1

            running_msg = " ".join(
                f"iter{k}={correct_per_iter[k]/n_candidates:.4f}" for k in range(K)
            )
            print(f"[L2L Eval Iter] batch {batch_idx} cum problems={n_problems} {running_msg} "
                  f"best={best_iter_correct/n_problems/n:.4f}")

        if n_problems == 0:
            print("[L2L Eval Iter] WARNING no val problems processed")
            return

        # Final metrics
        metrics = {f"eval/iter_{k}/acc": correct_per_iter[k] / n_candidates for k in range(K)}
        for k in range(1, K):
            metrics[f"eval/iter_{k}/parse_ok"] = parse_ok_per_iter[k] / n_candidates
        metrics["eval/best_iter_acc"] = best_iter_correct / (n_problems * n)
        metrics["eval/n_problems"] = n_problems
        metrics["eval/n_candidates"] = n_candidates
        metrics["eval/global_step"] = self.global_steps
        metrics["eval/iter_compact_steps"] = K
        metrics["eval/iter_compact_carry_prev"] = int(carry_prev)
        print(f"[L2L Eval Iter] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if eval_dump_dir is not None:
            try:
                summary_path = os.path.join(eval_dump_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "experiment_name": self.config.trainer.experiment_name,
                        "global_step": self.global_steps,
                        "n": n,
                        "K": K,
                        "carry_prev": carry_prev,
                        "eval_max_problems": eval_max_problems,
                        **metrics,
                    }, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[L2L Eval Iter] summary written to {summary_path}")
            except Exception as e:
                print(f"[L2L Eval Iter Dump] summary.json write failed: {e}")

    SELF_REFINE_PROMPT_TEMPLATE = (
        "Here is your previous attempt:\n{previous_response}\n\n"
        "Review your work, identify any errors or improvements, then solve "
        "the problem again:\n{problem}"
    )

    def _run_self_refine_eval(self, logger):
        """Self-refine baseline (eval-only, base model).

        Two-pass per problem:
          1. actor.generate(problem)                          → response_1
          2. actor.generate(REFINE(problem, response_1))      → response_2
        math_verify scores both. Compares the second-attempt accuracy against
        the first-attempt baseline to show whether "look at your own output
        and try again" by itself (no exp_learner) shifts performance.

        Aggregated metrics:
          - eval/acc_pass1_mean       : pass-1 mean accuracy (B*n samples)
          - eval/acc_pass2_mean       : pass-2 mean accuracy (B*n samples)
          - eval/acc_pass1_best_of_n  : ANY pass-1 rollout correct, per problem
          - eval/acc_pass2_best_of_n  : ANY pass-2 rollout correct, per problem
          - eval/delta_acc_mean       : pass2 - pass1 (mean)
          - eval/delta_acc_best_of_n  : pass2 - pass1 (best-of-n)
          - eval/n_problems / eval/n_candidates / eval/global_step

        Per-step dump dir contains pass1.jsonl, pass2.jsonl, summary.json.
        Row schema mirrors actor_grpo_eval's phase_a.jsonl, plus the pass-2
        rows carry refine_prompt + previous_response so a reader can see the
        full chain of (problem → resp1 → refine_prompt → resp2 → reward).
        """
        n = self.config.actor_rollout_ref.rollout.n
        eval_max_problems = int(self.config.trainer.get("eval_max_problems", 500))

        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval")
        pass1_path = os.path.join(eval_dump_dir, "pass1.jsonl") if eval_dump_dir else None
        pass2_path = os.path.join(eval_dump_dir, "pass2.jsonl") if eval_dump_dir else None
        for p in (pass1_path, pass2_path):
            if p and os.path.exists(p):
                os.remove(p)

        pass1_correct = 0.0
        pass2_correct = 0.0
        pass1_best_of_n = 0.0
        pass2_best_of_n = 0.0
        n_problems = 0
        n_candidates = 0

        val_iter = iter(self.val_dataloader)
        batch_idx = 0

        while n_problems < eval_max_problems:
            try:
                batch_dict = next(val_iter)
            except StopIteration:
                print(f"[self_refine Eval] val dataloader exhausted at {n_problems}/{eval_max_problems}")
                break

            batch = DataProto.from_single_dict(batch_dict)
            remaining = eval_max_problems - n_problems
            if len(batch) > remaining:
                batch = batch[:remaining]
            B = len(batch)
            source_raw_prompts = list(batch.non_tensor_batch["raw_prompt"])

            # ── Pass 1: plain generate, n rollouts per problem ─────────
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
            )
            gen_batch.non_tensor_batch.pop("raw_prompt_ids", None)
            gen_batch.non_tensor_batch.pop("raw_prompt", None)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
            }
            gen_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, self.actor_rollout_wg.world_size,
            )
            gen_padded.meta_info["n"] = n
            pass1_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
            pass1 = unpad_dataproto(pass1_padded, pad_size * n)

            src_indices = np.array([i // n for i in range(B * n)])
            for key in ("reward_model", "data_source"):
                if key in batch.non_tensor_batch:
                    pass1.non_tensor_batch[key] = batch.non_tensor_batch[key][src_indices]
            pass1_reward, _ = compute_reward(pass1, self.reward_fn)
            pass1_acc = (pass1_reward.sum(-1) == 1.0).float()  # (B*n,)
            pass1_correct += pass1_acc.sum().item()
            pass1_responses = [
                self.tokenizer.decode(pass1.batch["responses"][i], skip_special_tokens=True)
                for i in range(B * n)
            ]

            # ── Pass 2: refine prompt = (problem, pass-1 response) ─────
            refine_msgs_list = []
            refine_prompt_texts = []
            for i in range(B * n):
                src_idx = i // n
                problem = source_raw_prompts[src_idx][-1]["content"]
                refine_content = self.SELF_REFINE_PROMPT_TEMPLATE.format(
                    previous_response=pass1_responses[i], problem=problem,
                )
                refine_prompt_texts.append(refine_content)
                refine_msgs_list.append([{"role": "user", "content": refine_content}])
            refine_tokenized = [self.val_dataset.re_tokenize(m) for m in refine_msgs_list]
            refine_batch = DataProto.from_single_dict({
                "input_ids": torch.stack([t["input_ids"] for t in refine_tokenized]),
                "attention_mask": torch.stack([t["attention_mask"] for t in refine_tokenized]),
                "position_ids": torch.stack([t["position_ids"] for t in refine_tokenized]),
            })
            refine_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "n": 1,
            }
            refine_padded, refine_pad = pad_dataproto_to_divisor(
                refine_batch, self.actor_rollout_wg.world_size,
            )
            pass2_padded = self.actor_rollout_wg.generate_sequences(refine_padded)
            pass2 = unpad_dataproto(pass2_padded, refine_pad)

            # Score pass 2 — reuse src_indices since refine_batch has B*n rows in same order.
            for key in ("reward_model", "data_source"):
                if key in batch.non_tensor_batch:
                    pass2.non_tensor_batch[key] = batch.non_tensor_batch[key][src_indices]
            pass2_reward, _ = compute_reward(pass2, self.reward_fn)
            pass2_acc = (pass2_reward.sum(-1) == 1.0).float()
            pass2_correct += pass2_acc.sum().item()

            pass1_best_of_n += (pass1_acc.view(B, n).sum(-1) > 0).float().sum().item()
            pass2_best_of_n += (pass2_acc.view(B, n).sum(-1) > 0).float().sum().item()

            # ── Dump ──
            if pass1_path is not None:
                try:
                    rm = batch.non_tensor_batch.get("reward_model", None)
                    ds = batch.non_tensor_batch.get("data_source", None)
                    pass1_full_prompts = self._decode_prompts_batch(pass1.batch["prompts"])
                    pass2_full_prompts = self._decode_prompts_batch(pass2.batch["prompts"])
                    pass2_responses = [
                        self.tokenizer.decode(pass2.batch["responses"][i], skip_special_tokens=True)
                        for i in range(B * n)
                    ]
                    pass1_acc_list = pass1_acc.cpu().tolist()
                    pass2_acc_list = pass2_acc.cpu().tolist()

                    recs1, recs2 = [], []
                    for i in range(B * n):
                        src_idx = i // n
                        rollout_idx = i % n
                        gt = rm[src_idx].get("ground_truth") if rm is not None else None
                        ds_str = str(ds[src_idx]) if ds is not None else None
                        problem_text = source_raw_prompts[src_idx][-1]["content"]
                        recs1.append({
                            "batch_idx": batch_idx,
                            "problem_idx_in_batch": src_idx,
                            "rollout_idx": rollout_idx,
                            "global_problem_idx": n_problems + src_idx,
                            "prompt_text": problem_text,
                            "full_prompt": pass1_full_prompts[i],
                            "response_text": pass1_responses[i],
                            "ground_truth": gt,
                            "data_source": ds_str,
                            "correct": bool(pass1_acc_list[i] == 1.0),
                        })
                        recs2.append({
                            "batch_idx": batch_idx,
                            "problem_idx_in_batch": src_idx,
                            "rollout_idx": rollout_idx,
                            "global_problem_idx": n_problems + src_idx,
                            "prompt_text": problem_text,
                            "refine_prompt": refine_prompt_texts[i],
                            "previous_response": pass1_responses[i],
                            "full_prompt": pass2_full_prompts[i],
                            "response_text": pass2_responses[i],
                            "ground_truth": gt,
                            "data_source": ds_str,
                            "correct": bool(pass2_acc_list[i] == 1.0),
                        })
                    self._dump_jsonl(pass1_path, recs1, append=True)
                    self._dump_jsonl(pass2_path, recs2, append=True)
                except Exception as e:
                    print(f"[self_refine Eval Dump] dump failed: {e}")

            n_candidates += B * n
            n_problems += B
            batch_idx += 1
            print(
                f"[self_refine Eval] batch {batch_idx} cum problems={n_problems} "
                f"acc_pass1={pass1_correct/n_candidates:.4f} "
                f"acc_pass2={pass2_correct/n_candidates:.4f} "
                f"delta={(pass2_correct - pass1_correct)/n_candidates:+.4f}",
            )

        if n_problems == 0:
            print("[self_refine Eval] WARNING no val problems processed")
            return

        acc_pass1_mean = pass1_correct / n_candidates
        acc_pass2_mean = pass2_correct / n_candidates
        acc_pass1_best_of_n = pass1_best_of_n / n_problems
        acc_pass2_best_of_n = pass2_best_of_n / n_problems
        metrics = {
            "eval/acc_pass1_mean": acc_pass1_mean,
            "eval/acc_pass2_mean": acc_pass2_mean,
            "eval/acc_pass1_best_of_n": acc_pass1_best_of_n,
            "eval/acc_pass2_best_of_n": acc_pass2_best_of_n,
            "eval/delta_acc_mean": acc_pass2_mean - acc_pass1_mean,
            "eval/delta_acc_best_of_n": acc_pass2_best_of_n - acc_pass1_best_of_n,
            "eval/n_problems": n_problems,
            "eval/n_candidates": n_candidates,
            "eval/global_step": self.global_steps,
        }
        print(f"[self_refine Eval] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if eval_dump_dir is not None:
            try:
                summary_path = os.path.join(eval_dump_dir, "summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "experiment_name": self.config.trainer.experiment_name,
                        "global_step": self.global_steps,
                        "n": n,
                        "eval_max_problems": eval_max_problems,
                        **metrics,
                    }, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[self_refine Eval] summary written to {summary_path}")
            except Exception as e:
                print(f"[self_refine Eval Dump] summary.json write failed: {e}")

    def _run_self_refine_eval_textgame(self, logger):
        """Self-refine textgame baseline (eval-only, base model).

        Same shape as `_run_self_refine_eval` (math) but the unit of work is
        an env seed, not a problem string, and pass-1 trajectories are loaded
        from a Phase A cache (same files L2L textgame eval already consumes,
        so pass-1 acc is bit-for-bit comparable across baselines).

        Per (env_idx, rollout_idx):
          1. Pass 1 — load cached trajectory + final reward from
             `trainer.phase_a_cache_path`. No new rollout.
          2. Pass 2 — actor re-plays the env on the SAME seed as Phase A
             (so grid / start / goal are identical), with the pass-1
             trajectory injected as a "previous attempt" block in the
             rollout-worker prompt (see `_build_textgame_refine_prompt` in
             vllm_rollout_spmd.py). Binary 0/1 reward from env.

        Aggregated metrics (named to match math `_run_self_refine_eval`):
          - eval/acc_pass1_mean / eval/acc_pass2_mean
          - eval/acc_pass1_best_of_n / eval/acc_pass2_best_of_n
          - eval/delta_acc_mean / eval/delta_acc_best_of_n
          - eval/n_problems / eval/n_candidates / eval/global_step

        Dump dir layout (mirrors math self_refine):
          pass1.jsonl  — echo of cache rows (one per env×rollout)
          pass2.jsonl  — previous_attempt + pass-2 history + reward
          summary.json — metrics + env_id + held_out_size
        """
        from omegaconf import OmegaConf

        n = int(self.config.actor_rollout_ref.rollout.n)
        HELD_OUT_SIZE = int(self.config.trainer.get("held_out_size", 500))
        # Match L2L textgame eval's world-size alignment so the same cache
        # file (which was dumped with world_size=8) plugs in cleanly.
        HELD_OUT_SIZE = HELD_OUT_SIZE - HELD_OUT_SIZE % self.actor_rollout_wg.world_size
        assert HELD_OUT_SIZE > 0, "held_out_size after world_size rounding is 0"

        num_steps = int(self.config.trainer.textgame_max_steps)

        # Deterministic env seeds — identical formula to L2L textgame eval
        # so the cache and the env we rebuild here both reset on the same
        # seed → identical grid/start/goal between pass 1 and pass 2.
        eval_seeds = [468382021 + 78025 + i * 1000 for i in range(HELD_OUT_SIZE)]

        # ── Load Phase A cache ─────────────────────────────────────────
        cache_path = OmegaConf.select(self.config.trainer, "phase_a_cache_path", default=None)
        assert cache_path, "self_refine_eval_textgame requires trainer.phase_a_cache_path"
        by_key = {}
        with open(cache_path, "r", encoding="utf-8") as cf:
            for line in cf:
                r = json.loads(line)
                by_key[(int(r["env_idx"]), int(r["rollout_idx"]))] = r
        print(f"[self_refine Eval Textgame] loaded {len(by_key)} cache rows from {cache_path}")

        # Sanity: every (env_idx, rollout_idx) pair we'll need must exist
        # and the seed in the cache must match the deterministic formula.
        for env_idx in range(HELD_OUT_SIZE):
            for rollout_idx in range(n):
                r = by_key.get((env_idx, rollout_idx))
                if r is None:
                    raise ValueError(
                        f"Phase A cache missing (env_idx={env_idx}, rollout_idx={rollout_idx}); "
                        f"need {HELD_OUT_SIZE}×{n}={HELD_OUT_SIZE*n} rows."
                    )
                if int(r.get("seed", -1)) != eval_seeds[env_idx]:
                    raise ValueError(
                        f"Phase A cache seed mismatch at env_idx={env_idx}: "
                        f"cache={r.get('seed')}, expected={eval_seeds[env_idx]} "
                        f"(check held_out_size matches the cache dump)"
                    )

        # Pass-1 metrics + per-env trajectories.
        pass1_rewards = [0.0] * (HELD_OUT_SIZE * n)
        previous_attempts = [""] * (HELD_OUT_SIZE * n)
        for rollout_idx in range(n):
            for env_idx in range(HELD_OUT_SIZE):
                flat = rollout_idx * HELD_OUT_SIZE + env_idx
                r = by_key[(env_idx, rollout_idx)]
                pass1_rewards[flat] = float(r["reward"])
                # Compose the pass-1 trajectory string injected into pass-2
                # prompt: "Round{k}_Input/Output" tags match the L2L Phase B
                # prompt format so the model recognizes the layout. Final
                # feedback (Success/Failure + reason) appended at the end.
                history = r.get("history", [])
                lines = []
                for step_info in history:
                    step_num = step_info.get("step", 0)
                    obs = step_info.get("observation", "")
                    resp = step_info.get("response", "")
                    lines.append(f"Round{step_num}_Input: {obs}")
                    lines.append(f"Round{step_num}_Output: {resp}")
                feedback = r.get("final_feedback", "")
                previous_attempts[flat] = "\n\n".join(lines) + (f"\n\n{feedback}" if feedback else "")

        # ── Pass 2: rollout-major, one round per rollout_idx ───────────
        pass2_rewards = [0.0] * (HELD_OUT_SIZE * n)
        pass2_trajs = [None] * (HELD_OUT_SIZE * n)
        for rollout_idx in range(n):
            round_prev = previous_attempts[rollout_idx * HELD_OUT_SIZE:(rollout_idx + 1) * HELD_OUT_SIZE]
            out = self.actor_rollout_wg.generate_sequences_textgame(
                env_config=self.textgame_env_config,
                env_num=HELD_OUT_SIZE,
                tokenizer=self.tokenizer,
                experiences=[""] * HELD_OUT_SIZE,
                num_steps=num_steps,
                seeds=eval_seeds,
                validate=True,
                previous_attempts=round_prev,
            )
            if isinstance(out, list):
                out = out[0]
            trajs = out["env_trajectories"]
            rewards = [rd[0] for rd in out["reward_list"]]
            for env_idx in range(HELD_OUT_SIZE):
                flat = rollout_idx * HELD_OUT_SIZE + env_idx
                pass2_trajs[flat] = trajs.get(env_idx, {})
                pass2_rewards[flat] = float(rewards[env_idx])
            print(f"[self_refine Eval Textgame] pass-2 rollout {rollout_idx+1}/{n} done "
                  f"(running acc_pass2={sum(1 for r in pass2_rewards[:(rollout_idx+1)*HELD_OUT_SIZE] if r == 1.0)/((rollout_idx+1)*HELD_OUT_SIZE):.4f})")

        # ── Aggregate (env-major) ──────────────────────────────────────
        pass1_acc = torch.tensor(pass1_rewards, dtype=torch.float32)
        pass2_acc = torch.tensor(pass2_rewards, dtype=torch.float32)
        pass1_correct = float((pass1_acc == 1.0).sum().item())
        pass2_correct = float((pass2_acc == 1.0).sum().item())
        # rollout-major flat: i = rollout_idx * B + env_idx → reshape (n, B)
        # then transpose to (B, n) for per-env best-of-n.
        B = HELD_OUT_SIZE
        pass1_matrix = (pass1_acc == 1.0).float().view(n, B).t()
        pass2_matrix = (pass2_acc == 1.0).float().view(n, B).t()
        pass1_best_of_n = float((pass1_matrix.sum(dim=-1) > 0).float().sum().item())
        pass2_best_of_n = float((pass2_matrix.sum(dim=-1) > 0).float().sum().item())

        n_problems = B
        n_candidates = B * n
        acc_pass1_mean = pass1_correct / n_candidates
        acc_pass2_mean = pass2_correct / n_candidates
        acc_pass1_best_of_n = pass1_best_of_n / n_problems
        acc_pass2_best_of_n = pass2_best_of_n / n_problems
        metrics = {
            "eval/acc_pass1_mean": acc_pass1_mean,
            "eval/acc_pass2_mean": acc_pass2_mean,
            "eval/acc_pass1_best_of_n": acc_pass1_best_of_n,
            "eval/acc_pass2_best_of_n": acc_pass2_best_of_n,
            "eval/delta_acc_mean": acc_pass2_mean - acc_pass1_mean,
            "eval/delta_acc_best_of_n": acc_pass2_best_of_n - acc_pass1_best_of_n,
            "eval/n_problems": n_problems,
            "eval/n_candidates": n_candidates,
            "eval/global_step": self.global_steps,
        }
        print(f"[self_refine Eval Textgame] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        # ── Dump ───────────────────────────────────────────────────────
        eval_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval")
        if eval_dump_dir is None:
            return
        pass1_path = os.path.join(eval_dump_dir, "pass1.jsonl")
        pass2_path = os.path.join(eval_dump_dir, "pass2.jsonl")
        for p in (pass1_path, pass2_path):
            if os.path.exists(p):
                os.remove(p)
        try:
            recs1, recs2 = [], []
            for rollout_idx in range(n):
                for env_idx in range(HELD_OUT_SIZE):
                    flat = rollout_idx * HELD_OUT_SIZE + env_idx
                    cache_row = by_key[(env_idx, rollout_idx)]
                    recs1.append({
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "reward": pass1_rewards[flat],
                        "correct": bool(pass1_rewards[flat] == 1.0),
                        "n_steps": int(cache_row.get("n_steps", 0)),
                        "stop_reason": cache_row.get("stop_reason", "unknown"),
                        "history": cache_row.get("history", []),
                        "final_feedback": cache_row.get("final_feedback", ""),
                    })
                    traj = pass2_trajs[flat] or {}
                    history = traj.get("history", [])
                    recs2.append({
                        "env_idx": env_idx,
                        "rollout_idx": rollout_idx,
                        "seed": eval_seeds[env_idx],
                        "previous_attempt": previous_attempts[flat],
                        "reward": pass2_rewards[flat],
                        "correct": bool(pass2_rewards[flat] == 1.0),
                        "n_steps": len(history),
                        "stop_reason": traj.get("stop_reason", "unknown"),
                        "history": [
                            {"step": s.get("step", k),
                             "observation": s.get("current_step_observation", ""),
                             "response": s.get("raw_response", "")}
                            for k, s in enumerate(history)
                        ],
                        "final_feedback": traj.get("final_feedback", ""),
                    })
            self._dump_jsonl(pass1_path, recs1, append=False)
            self._dump_jsonl(pass2_path, recs2, append=False)
        except Exception as e:
            print(f"[self_refine Eval Textgame Dump] dump failed: {e}")

        try:
            summary_path = os.path.join(eval_dump_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_name": self.config.trainer.experiment_name,
                    "global_step": self.global_steps,
                    "n": n,
                    "held_out_size": HELD_OUT_SIZE,
                    "textgame_env_id": self.config.trainer.textgame_env_id,
                    "phase_a_cache_path": cache_path,
                    **metrics,
                }, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(f"[self_refine Eval Textgame] summary written to {summary_path}")
        except Exception as e:
            print(f"[self_refine Eval Textgame Dump] summary.json write failed: {e}")

