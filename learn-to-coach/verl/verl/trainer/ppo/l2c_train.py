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
"""L2C training and evaluation orchestration."""

import os
import random
import sys
from pprint import pprint

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.l2c_mode import L2CRewardScope
from verl.trainer.ppo.reward import compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


class L2CTrainingMixin:
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if hasattr(self, "l2c_mode"):
            compatibility = "legacy iter_compact_steps" if self.l2c_mode.uses_legacy_iterations else "canonical"
            print(
                "[L2C Mode] "
                f"reward_scope={self.l2c_mode.reward_scope.value} "
                f"coaching_rounds={self.l2c_mode.coaching_rounds} "
                f"actor_attempts={self.l2c_mode.actor_attempts} ({compatibility})"
            )

        if not self.is_textgame:
            if self.config.trainer.prompt_version == 'v3':
                EXPERIENCE_UPDATE_PROMPT = """
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction (the user's question and your answer):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning. The experience you generate will be directly appended to the previous experience.

After careful reasoning step by step, output the final additional experience.
"""

            elif self.config.trainer.prompt_version == 'v5':
                # v5 = same-problem reflection prompt for iterative-compact eval
                # (test-time scaling). Semantics flipped vs v3/v4:
                #   - PREVIOUS_EXPERIENCE is notes about THIS problem (not a
                #     cross-problem knowledge base).
                #   - Output REPLACES the notes; no "append". Keeps the carried
                #     state bounded across K iterations.
                #   - Asks for failure-mode critique + next-attempt direction +
                #     intermediate results — the signals that actually compound
                #     under repeated re-solve, instead of generic platitudes.
                # Parser must extract from the last "# Notes" anchor so chain-
                # of-thought reasoning doesn't leak into PREVIOUS_EXPERIENCE on
                # the next iter — see _parse_experience v5 branch.
                EXPERIENCE_UPDATE_PROMPT = """You are solving a problem. Below are your previous attempts and the notes you took while solving. Refine the notes so your next attempt is more likely to be correct.

Problem and your latest attempt:
{LATEST_EXPERIENCE}

Your existing notes on this problem (from earlier attempts):
# Notes
{PREVIOUS_EXPERIENCE}

Your task:
- Critique the latest attempt. Where did the reasoning go wrong, or what was left unverified? If the answer seems correct, what would make you more sure?
- Update the notes so they capture: (a) what you have already tried and why it did or did not work, (b) the most promising direction for the next attempt, (c) any intermediate results worth keeping (lemmas, simplifications, candidate answers with confidence).
- The notes will REPLACE the previous notes, not be appended. Keep them concise — they will be re-read at every attempt.

After reasoning step by step, output the final notes in exactly this format:

# Notes
- ...
- ...
"""

            elif self.config.trainer.prompt_version == 'v4':
                EXPERIENCE_UPDATE_PROMPT = """
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction (the user's question and your answer):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning.

Rules:
- The experience you generate MUST be formatted strictly as a markdown list where each item starts with "- EXPERIENCE ITEM:", one per line:
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- The experience you generate will be directly appended to the previous experience.
- The change should introduce a general, high-level, widely applicable insight, not a detail from the specific interaction. The updated experience must remain concise, structured, and meaningful.
- If the new insight conflicts with any previous experience item, you are can describe the conflict and provide a resolution in the new item.

After careful reasoning step by step, output the final result in exactly this format:

Additional Experience:
# Experience
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
"""


        else:
            if self.config.trainer.prompt_version == 'v3':
                EXPERIENCE_UPDATE_PROMPT = """You are an AI language model that continuously refines its internal experience.
Here is the interaction history (the game environment (input) and your response and action (output)):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the multi-round interaction history and the previous experience, generate experience for future learning. You should conduct a deep, comparative analysis to infer the game rules and the fundamental principles behind winning and losing. Using the interaction history and environment feedback, hypothesize the game rules and effective winning strategies, and organize these insights into experience items that help the player succeed in the game.

Rules:
- The experience you generate will be directly appended to the previous experience. Do not repeat the previous experience. Make sure the newly generated experience is different from the previous experience.
- Your generated experience should be possible rules, instructions or winning strategies for the game. The experience should be generally useful rather than only applicable for the current map (board).

After careful reasoning step by step, output the final additional experience.
"""
            elif self.config.trainer.prompt_version == 'v5':
                # v5 = same-game reflection prompt for iterative-compact eval
                # (test-time scaling). Mirrors the math v5 semantics, adapted
                # to the multi-round game interaction:
                #   - PREVIOUS_EXPERIENCE is notes about THIS game/map (not a
                #     cross-game knowledge base).
                #   - Output REPLACES the notes; no "append". Keeps the carried
                #     state bounded across K iterations.
                #   - Asks for failure-mode critique + confirmed map facts +
                #     next-attempt plan — the signals that compound under
                #     repeated re-play, instead of generic platitudes.
                # Parser-wise v5 falls through to default strip (see
                # _parse_experience), so no extra parsing branch is needed.
                # NOTE: wording MUST match l2l-math-4's textgame v5 verbatim
                # so vanilla-trained ckpts (textgame-l2l-q3-*-binreward-...-v5-vanilla)
                # see the same prompt distribution at eval as at training time.
                EXPERIENCE_UPDATE_PROMPT = """You are playing a game. Below is your latest playthrough and the notes you took while playing. Refine the notes so your next attempt is more likely to win.

Interaction history (the game environment (input) and your response and action (output)):
{LATEST_EXPERIENCE}

Your existing notes on this game (from earlier attempts):
# Notes
{PREVIOUS_EXPERIENCE}

Your task:
- Critique the latest playthrough. Which actions moved you toward or away from the goal, what feedback did the environment give, and where did the attempt fail?
- Update the notes so they capture: (a) confirmed facts about this map (goal location, hazards/walls, what each action does), (b) action sequences that worked or failed and why, (c) the most promising plan for the next attempt.
- The notes will REPLACE the previous notes, not be appended. Keep them concise — they will be re-read at every attempt.

After reasoning step by step, output the final notes in exactly this format:

# Notes
- ...
- ...
"""
            elif self.config.trainer.prompt_version == 'v4':
                EXPERIENCE_UPDATE_PROMPT = """You are an AI language model that continuously refines its internal experience.
Here is the interaction history (the game environment (input) and your response and action (output)):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the multi-round interaction history and the previous experience, generate experience for future learning. You should conduct a deep, comparative analysis to infer the game rules and the fundamental principles behind winning and losing. Using the interaction history and environment feedback, hypothesize the game rules and effective winning strategies, and organize these insights into 1-2 concise, high-level, and widely applicable experience items that help the player succeed in the game.

Rules:
- The experience you generate MUST be formatted strictly as a markdown item which starts with "- EXPERIENCE ITEM:":
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- The experience you generate will be directly appended to the previous experience. Do not repeat the previous experience. Make sure the newly generated experience is different from the previous experience.
- Your generated experience should be possible rules, instructions or winning strategies for the game. The experience should be generally useful rather than only applicable for the current map (board).

After careful reasoning step by step, output the final result in exactly this format:

Additional Experience (Rules or Strategies):
# Experience
- EXPERIENCE ITEM: ...
"""
        
        if self.config.trainer.prompt_version == 'v5':
            # v5 = same-problem reflection. Don't call it "the new problem"
            # (it's the same problem being re-solved), don't ask the model to
            # cite which experience item it uses (the notes are about this
            # problem — they all apply).
            EXPERIENCE_SOLVE_PROMPT_TEMPLATE = """You have been working on the problem below. Here are your notes from previous attempts; use them to avoid past mistakes and find the correct answer.

# Notes
{experience}

Solve the problem. You may continue from a promising direction in the notes or start over if needed; the notes are guidance, not constraints.

Problem:
{prompt}"""
        else:
            EXPERIENCE_SOLVE_PROMPT_TEMPLATE = """Given previous learned experience:
# Experience
{experience}

Solve the new problem and explain what part of experience you use and how you use it in the reasoning process:
{prompt}"""


        self.experience_update_prompt = EXPERIENCE_UPDATE_PROMPT
        self.experience_solve_prompt_template = EXPERIENCE_SOLVE_PROMPT_TEMPLATE

        self_refine_runners = {
            "self_refine_eval": self._run_self_refine_eval,
            "self_refine_eval_textgame": self._run_self_refine_eval_textgame,
        }
        if self.config.trainer.stage in self_refine_runners:
            self_refine_runners[self.config.trainer.stage](logger)
            print("[Self-Refinement Eval] complete", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os.sync()
            os._exit(0)

        # ── L2L eval: load exp_learner from a specific ckpt, run aligned 3-phase
        # evaluation on val set, log metrics, and exit. Skips val_before_train
        # and the training loop entirely.
        if self.config.trainer.stage == "l2l_eval":
            phase_a_dump_only = bool(
                OmegaConf.select(self.config.trainer, "phase_a_dump_only", default=False)
            )
            if phase_a_dump_only:
                # Cache schema is reward-scope independent. Use the aligned
                # same-instance Phase A path and stop before experience generation.
                if self.is_textgame:
                    self._run_l2l_eval_textgame(logger)
                else:
                    self._run_l2l_eval(logger)
            elif self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                if self.is_textgame:
                    self._run_l2c_cross_instance_eval_textgame(logger)
                else:
                    self._run_l2c_cross_instance_eval_math(logger)
            elif self.is_textgame:
                if self.l2c_mode.coaching_rounds > 1:
                    self._run_l2l_eval_iterative_textgame(
                        logger, self.l2c_mode.actor_attempts
                    )
                else:
                    self._run_l2l_eval_textgame(logger)
            else:
                if self.l2c_mode.coaching_rounds > 1:
                    self._run_l2l_eval_iterative(logger, self.l2c_mode.actor_attempts)
                else:
                    self._run_l2l_eval(logger)
            # Bypass Ray's shutdown segfault (KillActor -> TaskEventBufferImpl::
            # FlushEvents crashes workers, leaving driver hanging on ack). All
            # business-state (summary.json + dumps) is already on blob by now;
            # container cleanup will reclaim worker actors.
            print("[L2L Eval] complete, os.sync + os._exit(0) to bypass Ray shutdown hang", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os.sync()  # flush all dirty pages incl. blob fuse buffer
            os._exit(0)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        EXPERIENCE = "No previous experience."
        self.experience = EXPERIENCE

        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter([None]) if self.is_textgame else self.train_dataloader
            for batch_dict in dataloader_iter:
                
                if self.config.trainer.stage == "l2l" and self.is_textgame:
                    if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                        metrics = {}
                        timing_raw = {}
                        is_last_step = self.global_steps >= self.total_training_steps
                        self._run_l2c_cross_instance_train_textgame_step(
                            n=int(self.config.actor_rollout_ref.rollout.n),
                            metrics=metrics,
                            timing_raw=timing_raw,
                        )
                        metrics.update(
                            {
                                "training/global_step": self.global_steps,
                                "training/epoch": epoch,
                                "training/coaching_round": 1,
                            }
                        )
                        metrics.update(
                            {f"timing_s/{key}": value for key, value in timing_raw.items()}
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            self._save_checkpoint()
                        progress_bar.update(1)
                        self.global_steps += 1
                        if is_last_step:
                            pprint(
                                f"[L2C Cross Textgame] Training complete at step "
                                f"{self.global_steps - 1}"
                            )
                            progress_bar.close()
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os.sync()
                            os._exit(0)
                        continue

                    # ══════════════════════════════════════════════════════
                    # L2L Textgame Training Loop — K-iter same-env refinement
                    # ─────────────────────────────────────────────────────
                    # Analog of the math K-iter loop below, but rollouts go
                    # through generate_sequences_textgame (env stepping) and
                    # per-env state (PREV exp + LATEST traj) rolls forward.
                    # Fixed seeds across all K iters within a step (Plan A —
                    # same envs refined K times, matching the semantics of
                    # _run_l2l_eval_iterative_textgame).
                    #
                    #   iter 0 setup: bare actor plays each env with empty exp
                    #                 → LATEST_TRAJ_PER_ENV = traj_0 (baseline)
                    #   iter k = 0..K-1 (each = 1 grad update / 1 global_step):
                    #     exp_learner((history_{k-1}), PREV=EXP_PER_ENV[env])
                    #       → n cands per env (B*n exps)
                    #     actor RE-PLAYS each env n times (SAME seeds) with the
                    #       n candidate exps → B*n rewards
                    #     GRPO update on exp_learner (per-env n-group advantage)
                    #     carry idx-0 forward: EXP_PER_ENV = experiences[env*n]
                    #                          LATEST_TRAJ_PER_ENV = phase_c_trajs[env*n]
                    #
                    # K=1 fallback = vanilla single-step textgame training
                    # (Phase A → B → C → grad update, one grad per outer batch).
                    # ══════════════════════════════════════════════════════

                    n = int(self.config.actor_rollout_ref.rollout.n)
                    K = self.l2c_mode.coaching_rounds

                    B = int(self.config.data.train_batch_size)
                    # Same seed formula as the OPCD-style textgame training path
                    # (L3959) so a given (global_step, oel_round) → deterministic
                    # env grid regardless of K.
                    seeds = [
                        505019424 + 90039 + 100000 + (self.config.trainer.oel_round - 1) * 10000000 + tmp_num * 1000
                        for tmp_num in range(self.global_steps * B, (self.global_steps + 1) * B)
                    ]

                    num_steps = int(self.config.trainer.textgame_max_steps)
                    max_exp_tokens = int(self.config.trainer.experience_max_length)

                    def _traj_to_history_text(traj):
                        return self._l2c_textgame_history(traj or {})

                    # ── iter-0 baseline: bare actor rollout on each env ──────
                    # Runs once at the start of the step (not per iter). Fills
                    # LATEST_TRAJ_PER_ENV so iter 0's exp_learner has a history
                    # to reflect on.
                    empty_exps = [""] * B
                    phase_a_out = self.actor_rollout_wg.generate_sequences_textgame(
                        env_config=self.textgame_env_config,
                        env_num=B,
                        tokenizer=self.tokenizer,
                        experiences=empty_exps,
                        num_steps=num_steps,
                        seeds=seeds,
                        validate=True,
                    )
                    if isinstance(phase_a_out, list):
                        phase_a_out = phase_a_out[0]
                    phase_a_trajs = phase_a_out["env_trajectories"]
                    phase_a_rewards = [float(rd[0]) for rd in phase_a_out["reward_list"]]
                    LATEST_TRAJ_PER_ENV = [phase_a_trajs.get(env_idx, {}) for env_idx in range(B)]
                    EXP_PER_ENV = [""] * B

                    for iter_k in range(K):
                        metrics = {}
                        timing_raw = {}
                        is_last_step = self.global_steps >= self.total_training_steps

                        with marked_timer("step", timing_raw):
                            # ── Phase B: exp_learner emits n candidates per env ──
                            with marked_timer("phase_b_gen", timing_raw, color="orange"):
                                # Snapshot PREV before roll-forward so per-iter
                                # dumps show what actually went into exp_learner
                                # this iter (EXP_PER_ENV gets overwritten below).
                                prev_experiences_snapshot = list(EXP_PER_ENV)
                                exp_learner_prompts = []
                                latest_texts = []
                                for env_idx in range(B):
                                    latest = _traj_to_history_text(LATEST_TRAJ_PER_ENV[env_idx])
                                    latest_texts.append(latest)
                                    prev = EXP_PER_ENV[env_idx] if EXP_PER_ENV[env_idx] else "No previous experience."
                                    exp_learner_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                                        PREVIOUS_EXPERIENCE=prev,
                                        LATEST_EXPERIENCE=latest,
                                    ))

                                dbg_idx = random.randint(0, B - 1)
                                dbg_prompt = exp_learner_prompts[dbg_idx]
                                print(f"[L2L EXP INPUT DEBUG] step={self.global_steps} iter_k={iter_k} idx={dbg_idx} len_chars={len(dbg_prompt)}")
                                print(f"  head 800: {dbg_prompt[:800]!r}")
                                print(f"  tail 400: {dbg_prompt[-400:]!r}")

                                exp_learner_batch = self._l2c_tokenize_textgame_prompts(
                                    exp_learner_prompts
                                )
                                exp_learner_batch.meta_info = {
                                    "eos_token_id": self.tokenizer.eos_token_id,
                                    "pad_token_id": self.tokenizer.pad_token_id,
                                    "recompute_log_prob": False,
                                    "do_sample": True,
                                    "validate": True,
                                    "n": n,   # n candidate exps per env → B*n outputs
                                }
                                exp_batch_padded, exp_pad_size = pad_dataproto_to_divisor(
                                    exp_learner_batch, self.exp_learner_wg.world_size
                                )
                                exp_output_padded = self.exp_learner_wg.generate_sequences(exp_batch_padded)
                                exp_output = unpad_dataproto(exp_output_padded, exp_pad_size * n)

                                experiences = []
                                raw_exp_texts = []
                                parsed_exps = []
                                parse_oks = []
                                parse_success_count = 0
                                sample_idx = random.randint(0, B * n - 1)
                                failed_dumped = False
                                for i in range(B * n):
                                    env_idx = i // n
                                    exp_text = self.tokenizer.decode(
                                        exp_output.batch["responses"][i], skip_special_tokens=True
                                    )
                                    raw_exp_texts.append(exp_text)
                                    parsed = self._parse_experience(exp_text)
                                    parsed_exps.append(parsed)
                                    ok = bool(parsed)
                                    parse_oks.append(ok)
                                    if ok:
                                        parse_success_count += 1

                                    if i == sample_idx:
                                        print(f"[L2L EXP DEBUG] step={self.global_steps} iter_k={iter_k} prompt_version={self.config.trainer.prompt_version} parsed_len={len(parsed)}")
                                        print(f"  raw exp_text head: {exp_text[:300]!r}")
                                        print(f"  raw exp_text tail: {exp_text[-200:]!r}")
                                        print(f"  parsed head: {parsed[:200]!r}")
                                    if not parsed and not failed_dumped:
                                        failed_dumped = True
                                        print(f"[L2L EXP FAIL DEBUG] step={self.global_steps} iter_k={iter_k} idx={i} exp_text_len={len(exp_text)}")
                                        print(f"  raw exp_text head: {exp_text[:500]!r}")
                                        print(f"  raw exp_text tail: {exp_text[-300:]!r}")

                                    # REPLACE semantics; on parse fail keep PREV
                                    # (matches math K-iter L4749-4756 and
                                    # _run_l2l_eval_iterative_textgame L1995-1999).
                                    combined = parsed if parsed else EXP_PER_ENV[env_idx]
                                    combined = self._truncate_experience(combined, max_exp_tokens) if combined else ""
                                    experiences.append(combined)

                                metrics["experience/parse_success_rate"] = parse_success_count / max(B * n, 1)

                            # ── Phase C: actor re-plays each env n times ──
                            # env-major layout: i = env_idx * n + rollout_idx, so
                            # experiences[env_idx * n + r] pairs with env_idx at
                            # rollout r. Matches meta_info['n']=n's src_idx=i//n.
                            with marked_timer("phase_c_resolve", timing_raw, color="red"):
                                phase_c_rewards = [0.0] * (B * n)
                                phase_c_trajs = [None] * (B * n)
                                for r in range(n):
                                    round_exps = [experiences[env_idx * n + r] for env_idx in range(B)]
                                    out = self.actor_rollout_wg.generate_sequences_textgame(
                                        env_config=self.textgame_env_config,
                                        env_num=B,
                                        tokenizer=self.tokenizer,
                                        experiences=round_exps,
                                        num_steps=num_steps,
                                        seeds=seeds,   # Plan A: same seeds every iter+rollout
                                        validate=True,
                                    )
                                    if isinstance(out, list):
                                        out = out[0]
                                    trajs = out["env_trajectories"]
                                    rewards = [rd[0] for rd in out["reward_list"]]
                                    for env_idx in range(B):
                                        flat = env_idx * n + r
                                        phase_c_trajs[flat] = trajs.get(env_idx, {})
                                        phase_c_rewards[flat] = float(rewards[env_idx])

                                reward_tensor = torch.tensor(phase_c_rewards, dtype=torch.float32)
                                metrics["l2l/train_acc_no_exp"] = sum(1.0 for r in phase_a_rewards if r == 1.0) / max(B, 1)
                                metrics["l2l/train_acc_with_exp_mean"] = reward_tensor.mean().item()

                            # ── GRPO update on exp_learner ──
                            # Advantage = per-env normalization over the n
                            # candidate exps: (r_i - mean_env) / (std_env + eps).
                            # Same recipe as math K-iter L4767-4798.
                            with marked_timer("update_exp_learner", timing_raw, color="green"):
                                reward_matrix = reward_tensor.view(B, n)
                                group_mean = reward_matrix.mean(dim=-1, keepdim=True)
                                group_std = reward_matrix.std(dim=-1, keepdim=True)
                                adv_matrix = (reward_matrix - group_mean) / (group_std + 1e-8)
                                exp_advantages = adv_matrix.view(-1)   # (B*n,)

                                response_mask_exp = exp_output.batch["attention_mask"][:, exp_learner_batch.batch["input_ids"].shape[-1]:]
                                resp_len = exp_output.batch["responses"].shape[-1]
                                response_mask_exp = response_mask_exp[:, :resp_len]
                                exp_output.batch["advantages"] = exp_advantages.unsqueeze(-1).to(response_mask_exp.device) * response_mask_exp

                                exp_output.meta_info["recompute_log_prob"] = True
                                exp_compute_padded, exp_compute_pad = pad_dataproto_to_divisor(
                                    exp_output, self.exp_learner_wg.world_size
                                )
                                old_log_prob_output = self.exp_learner_wg.compute_log_prob(exp_compute_padded)
                                old_log_prob_output = unpad_dataproto(old_log_prob_output, exp_compute_pad)
                                exp_output.batch["old_log_probs"] = old_log_prob_output.batch["old_log_probs"]

                                exp_output.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                exp_output.meta_info["l2l_mode"] = True
                                exp_output.meta_info["global_token_num"] = torch.sum(
                                    exp_output.batch["attention_mask"], dim=-1
                                ).tolist()
                                exp_update_padded, exp_update_pad = pad_dataproto_to_divisor(
                                    exp_output, self.exp_learner_wg.world_size
                                )
                                actor_output = self.exp_learner_wg.update_actor(exp_update_padded)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update({f"exp_learner/{k_m}": v for k_m, v in actor_output_metrics.items()})

                                all_exp_lens = [
                                    len(self.tokenizer.encode(e, add_special_tokens=False))
                                    for e in experiences
                                ]
                                metrics["reward/mean_correct"] = reward_matrix.mean().item()
                                metrics["reward/group_std_mean"] = group_std.mean().item()
                                metrics["experience/length_tokens"] = sum(all_exp_lens) / max(len(all_exp_lens), 1)

                            # ── Roll per-env state forward (idx 0 of each env) ──
                            EXP_PER_ENV = [experiences[env_idx * n] for env_idx in range(B)]
                            LATEST_TRAJ_PER_ENV = [phase_c_trajs[env_idx * n] for env_idx in range(B)]

                            # ── Per-iter dump ──
                            step_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}")
                            if step_dump_dir is not None:
                                try:
                                    if iter_k == 0:
                                        phase_a_records = []
                                        for env_idx in range(B):
                                            traj = phase_a_trajs.get(env_idx, {})
                                            history = traj.get("history", [])
                                            phase_a_records.append({
                                                "env_idx": env_idx,
                                                "iter_k": 0,
                                                "seed": seeds[env_idx],
                                                "reward": phase_a_rewards[env_idx],
                                                "correct": bool(phase_a_rewards[env_idx] == 1.0),
                                                "n_steps": len(history),
                                                "stop_reason": traj.get("stop_reason", "unknown"),
                                                "history": [
                                                    {"step": s["step"],
                                                     "observation": s.get("current_step_observation", ""),
                                                     "response": s.get("raw_response", "")}
                                                    for s in history
                                                ],
                                            })
                                        self._dump_jsonl(os.path.join(step_dump_dir, "phase_a.jsonl"), phase_a_records)

                                    phase_b_gen_records = []
                                    for i in range(B * n):
                                        env_idx = i // n
                                        rollout_idx = i % n
                                        phase_b_gen_records.append({
                                            "candidate_idx": i,
                                            "env_idx": env_idx,
                                            "rollout_idx": rollout_idx,
                                            "iter_k": iter_k,
                                            "seed": seeds[env_idx],
                                            "prev_experience": prev_experiences_snapshot[env_idx],
                                            "latest_history": latest_texts[env_idx],
                                            "exp_learner_input": exp_learner_prompts[env_idx],
                                            "raw_output": raw_exp_texts[i],
                                            "parsed_exp": parsed_exps[i],
                                            "parse_ok": parse_oks[i],
                                            "combined_truncated": experiences[i],
                                        })
                                    self._dump_jsonl(os.path.join(step_dump_dir, "phase_b_exp_gen.jsonl"), phase_b_gen_records)

                                    phase_c_records = []
                                    for i in range(B * n):
                                        env_idx = i // n
                                        rollout_idx = i % n
                                        traj = phase_c_trajs[i] or {}
                                        history = traj.get("history", [])
                                        phase_c_records.append({
                                            "candidate_idx": i,
                                            "env_idx": env_idx,
                                            "rollout_idx": rollout_idx,
                                            "iter_k": iter_k,
                                            "seed": seeds[env_idx],
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
                                    self._dump_jsonl(os.path.join(step_dump_dir, "phase_b_resolve.jsonl"), phase_c_records)
                                except Exception as e:
                                    print(f"[L2L Dump Textgame] step {self.global_steps} iter_k {iter_k} dump failed: {e}")

                        # ── Per-iter log + save + tick ──
                        metrics.update({
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                            "training/iter_k": iter_k,
                        })
                        metrics.update({f"timing_s/{k_t}": v for k_t, v in timing_raw.items()})
                        logger.log(data=metrics, step=self.global_steps)

                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            self._save_checkpoint()

                        progress_bar.update(1)
                        self.global_steps += 1

                        if is_last_step:
                            pprint(f"[L2L Textgame] Training complete at step {self.global_steps - 1}")
                            progress_bar.close()
                            print("[L2L Textgame] training complete, os.sync + os._exit(0) to bypass Ray shutdown hang", flush=True)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os.sync()
                            os._exit(0)


                elif self.config.trainer.stage == "l2l":
                    # ══════════════════════════════════════════════
                    # L2L Math Training Loop — K-iter same-problem refinement
                    # ──────────────────────────────────────────────
                    # Each dataloader batch (Q1..QB) is reused for K iters,
                    # mirroring _run_l2l_eval_iterative. Per-source state
                    # (PREV experience + LATEST response) rolls forward via
                    # the index-0 candidate of each src group.
                    #
                    #   iter 0:  bare actor.generate (no exp prepend) → A_0
                    #            exp_learner(Q, A_0, PREV="") → n cands
                    #            resolve, GRPO update; carry idx0 → next iter
                    #   iter k≥1: skip Phase A; reuse last iter's resolve idx0
                    #            as A_k. exp_learner(Q, A_k, PREV=E_{k-1}^0)
                    #            → n cands → resolve → GRPO update; carry idx0
                    #
                    # Each iter is one global_step (separate grad update + log).
                    # ══════════════════════════════════════════════

                    n = self.config.actor_rollout_ref.rollout.n
                    exp_learner_batch_size = int(self.config.trainer.get("exp_learner_batch_size", 1))
                    if exp_learner_batch_size < 1:
                        raise ValueError(
                            "trainer.exp_learner_batch_size must be positive for L2C math training"
                        )
                    K = self.l2c_mode.coaching_rounds
                    cross_dump_enabled = (
                        self.l2c_mode.reward_scope
                        is L2CRewardScope.CROSS_INSTANCE
                        and bool(self.config.trainer.get("dump_dir", None))
                    )

                    # ── Build per-batch shared state (computed once, reused K times) ──
                    batch = DataProto.from_single_dict(batch_dict)
                    probe_batch = None
                    if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                        probe_size = int(self.config.trainer.get("probe_size", 8))
                        if probe_size < 1:
                            raise ValueError(
                                "trainer.probe_size must be positive for cross-instance training"
                            )
                        required_batch_size = exp_learner_batch_size + probe_size
                        if len(batch) < required_batch_size:
                            raise ValueError(
                                f"cross-instance math training needs data.train_batch_size >= "
                                f"exp_learner_batch_size + probe_size = {required_batch_size}; "
                                f"received {len(batch)}"
                            )
                        exp_batch = batch[:exp_learner_batch_size]
                        probe_batch = batch[
                            exp_learner_batch_size:required_batch_size
                        ]
                    elif exp_learner_batch_size < len(batch):
                        exp_batch = batch[:exp_learner_batch_size]
                    else:
                        exp_batch = batch

                    # raw_prompt is reused every iter to build re-solve prompts;
                    # snapshot before any pop. Pop input_ids/etc once into
                    # gen_batch_template — iter 0 Phase A reuses these tensors
                    # directly (raw Q, no prepend since EXPERIENCE per-src = "").
                    source_raw_prompts = list(exp_batch.non_tensor_batch['raw_prompt'])
                    B = len(exp_batch)

                    # Per-source carried state — length B (not B*n; only idx0
                    # of each src's n candidates rolls forward).
                    EXP_PER_SRC = [""] * B            # PREV fed into exp_learner at iter k
                    LATEST_PER_SRC = [None] * B       # A_{k-1}; filled by iter 0 Phase A

                    for iter_k in range(K):
                        metrics = {}
                        timing_raw = {}
                        is_last_step = self.global_steps >= self.total_training_steps

                        with marked_timer("step", timing_raw):

                            # ── iter 0 only: Phase A (bare actor, no exp prepend) ──
                            if iter_k == 0:
                                with marked_timer("phase_a", timing_raw, color="blue"):
                                    gen_batch = exp_batch.pop(
                                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                                        non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
                                    )
                                    original_prompts = [
                                        source_raw_prompts[i][-1]['content'] for i in range(B)
                                    ]
                                    gen_batch.non_tensor_batch.pop("raw_prompt_ids", None)
                                    gen_batch.non_tensor_batch.pop("raw_prompt", None)
                                    gen_batch.meta_info = {
                                        "eos_token_id": self.tokenizer.eos_token_id,
                                        "pad_token_id": self.tokenizer.pad_token_id,
                                        "recompute_log_prob": False,
                                        "do_sample": True,
                                        "validate": True,
                                    }
                                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                                        gen_batch, self.actor_rollout_wg.world_size
                                    )
                                    gen_batch_padded.meta_info["n"] = 1
                                    gen_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                                    gen_output = unpad_dataproto(gen_output_padded, pad_size)

                                    eval_batch = exp_batch.union(gen_output)
                                    phase_a_reward_tensor, _ = compute_reward(eval_batch, self.reward_fn)
                                    train_rewards = phase_a_reward_tensor.sum(-1).cpu().tolist()
                                    train_acc = sum(1 for r in train_rewards if r == 1.0) / len(train_rewards)
                                    metrics["l2l/train_acc"] = train_acc

                                    phase_a_responses_text = [
                                        self.tokenizer.decode(gen_output.batch["responses"][i], skip_special_tokens=True)
                                        for i in range(len(gen_output))
                                    ]
                                    phase_a_full_prompts = self._decode_prompts_batch(gen_output.batch["prompts"])
                                    # Seed LATEST for iter 0's Phase B.
                                    LATEST_PER_SRC = list(phase_a_responses_text)

                            # ── Phase B: exp_learner generates n candidates per src ──
                            with marked_timer("phase_b_gen", timing_raw, color="orange"):
                                exp_learner_prompts = []
                                for i in range(B):
                                    latest = f"Input: {original_prompts[i]}\nOutput: {LATEST_PER_SRC[i]}"
                                    prev = EXP_PER_SRC[i] if EXP_PER_SRC[i] else "No previous experience."
                                    exp_learner_prompts.append(EXPERIENCE_UPDATE_PROMPT.format(
                                        PREVIOUS_EXPERIENCE=prev,
                                        LATEST_EXPERIENCE=latest,
                                    ))

                                dbg_idx = random.randint(0, B - 1)
                                dbg_prompt = exp_learner_prompts[dbg_idx]
                                print(f"[L2L EXP INPUT DEBUG] step={self.global_steps} iter_k={iter_k} idx={dbg_idx} len_chars={len(dbg_prompt)}")
                                print(f"  head 800: {dbg_prompt[:800]!r}")
                                print(f"  tail 400: {dbg_prompt[-400:]!r}")

                                exp_tokenized_list = [
                                    self.train_dataset.re_tokenize([{"role": "user", "content": p}])
                                    for p in exp_learner_prompts
                                ]
                                exp_learner_batch = DataProto.from_single_dict({
                                    "input_ids": torch.stack([t["input_ids"] for t in exp_tokenized_list]),
                                    "attention_mask": torch.stack([t["attention_mask"] for t in exp_tokenized_list]),
                                    "position_ids": torch.stack([t["position_ids"] for t in exp_tokenized_list]),
                                })
                                exp_learner_batch.meta_info = {
                                    "eos_token_id": self.tokenizer.eos_token_id,
                                    "pad_token_id": self.tokenizer.pad_token_id,
                                    "recompute_log_prob": False,
                                    "do_sample": True,
                                    "validate": True,
                                    "n": n,
                                }
                                exp_batch_padded, exp_pad_size = pad_dataproto_to_divisor(
                                    exp_learner_batch, self.exp_learner_wg.world_size
                                )
                                exp_output_padded = self.exp_learner_wg.generate_sequences(exp_batch_padded)
                                exp_output = unpad_dataproto(exp_output_padded, exp_pad_size * n)

                                # Parse all B*n candidates; combine = REPLACE
                                # (no append). On parse fail fall back to PREV
                                # so a noisy iter doesn't wipe accumulated state.
                                experiences = []
                                raw_exp_texts = []
                                parsed_exps = []
                                parse_oks = []
                                parse_success_count = 0
                                failed_idxs = []
                                sample_idx = random.randint(0, B * n - 1)
                                failed_dumped = False
                                for i in range(B * n):
                                    src_idx = i // n
                                    exp_text = self.tokenizer.decode(
                                        exp_output.batch["responses"][i], skip_special_tokens=True
                                    )
                                    raw_exp_texts.append(exp_text)
                                    pre_think = exp_text
                                    if "</think>" in pre_think:
                                        pre_think = pre_think.split("</think>")[-1]
                                    if self.config.trainer.prompt_version == 'v1':
                                        marker = "- EXPERIENCE ITEM:"
                                        lines = [l for l in pre_think.split("\n") if marker in l]
                                        result_lines = []
                                        for l in lines:
                                            parts = l.split(marker)[1:]
                                            result_lines.extend([marker + p.rstrip() for p in parts if p.strip()])
                                        parsed = "\n".join(result_lines)
                                    elif self.config.trainer.prompt_version == 'v2':
                                        lines = [l.strip() for l in pre_think.split("\n") if l.strip().startswith("- ")]
                                        parsed = "\n".join(lines)
                                    else:
                                        parsed = pre_think.strip()

                                    if parsed:
                                        parse_success_count += 1
                                    else:
                                        failed_idxs.append(i)
                                    parsed_exps.append(parsed)
                                    parse_oks.append(bool(parsed))

                                    if i == sample_idx:
                                        print(f"[L2L EXP DEBUG] step={self.global_steps} iter_k={iter_k} prompt_version={self.config.trainer.prompt_version} parsed_len={len(parsed)}")
                                        print(f"  raw exp_text head: {pre_think[:300]!r}")
                                        print(f"  raw exp_text tail: {pre_think[-200:]!r}")
                                        print(f"  parsed head: {parsed[:200]!r}")
                                    if not parsed and not failed_dumped:
                                        failed_dumped = True
                                        print(f"[L2L EXP FAIL DEBUG] step={self.global_steps} iter_k={iter_k} idx={i} exp_text_len={len(pre_think)}")
                                        print(f"  raw exp_text head: {pre_think[:500]!r}")
                                        print(f"  raw exp_text tail: {pre_think[-300:]!r}")

                                    # REPLACE semantics: prefer new parsed; on
                                    # fail keep PREV (don't reset to empty —
                                    # matches eval iter behavior at 1957).
                                    combined = parsed if parsed else EXP_PER_SRC[src_idx]
                                    combined = self._truncate_experience(
                                        combined, self.config.trainer.experience_max_length
                                    )
                                    experiences.append(combined)
                                metrics["experience/parse_success_rate"] = parse_success_count / max(B * n, 1)
                                metrics["experience/n_parse_failed"] = len(failed_idxs)

                            # ── Phase C: actor re-solves each candidate (B*n) ──
                            with marked_timer("phase_c_resolve", timing_raw, color="red"):
                                if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                                    if cross_dump_enabled:
                                        (
                                            reward_tensor,
                                            cross_pair_rewards,
                                            resolve_responses_text,
                                            resolve_full_prompts,
                                        ) = self._compute_cross_instance_reward_math(
                                            experiences,
                                            probe_batch,
                                            return_dump_info=True,
                                        )
                                    else:
                                        reward_tensor = (
                                            self._compute_cross_instance_reward_math(
                                                experiences,
                                                probe_batch,
                                            )
                                        )
                                else:
                                    (
                                        reward_tensor,
                                        resolve_responses_text,
                                        resolve_full_prompts,
                                        _,
                                    ) = self._compute_binary_reward_math(
                                        experiences,
                                        exp_batch,
                                        source_raw_prompts,
                                        n,
                                        return_dump_info=True,
                                    )

                            # ── GRPO update on exp_learner ──
                            with marked_timer("update_exp_learner", timing_raw, color="green"):
                                reward_matrix = reward_tensor.view(B, n)
                                group_mean = reward_matrix.mean(dim=-1, keepdim=True)
                                group_std = reward_matrix.std(dim=-1, keepdim=True)
                                adv_matrix = (reward_matrix - group_mean) / (group_std + 1e-8)
                                exp_advantages = adv_matrix.view(-1)

                                response_mask_exp = exp_output.batch["attention_mask"][:, exp_learner_batch.batch["input_ids"].shape[-1]:]
                                resp_len = exp_output.batch["responses"].shape[-1]
                                response_mask_exp = response_mask_exp[:, :resp_len]
                                exp_output.batch["advantages"] = exp_advantages.unsqueeze(-1).to(response_mask_exp.device) * response_mask_exp

                                exp_output.meta_info["recompute_log_prob"] = True
                                exp_compute_padded, exp_compute_pad = pad_dataproto_to_divisor(
                                    exp_output, self.exp_learner_wg.world_size
                                )
                                old_log_prob_output = self.exp_learner_wg.compute_log_prob(exp_compute_padded)
                                old_log_prob_output = unpad_dataproto(old_log_prob_output, exp_compute_pad)
                                exp_output.batch["old_log_probs"] = old_log_prob_output.batch["old_log_probs"]

                                exp_output.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                exp_output.meta_info["l2l_mode"] = True
                                exp_output.meta_info["global_token_num"] = torch.sum(
                                    exp_output.batch["attention_mask"], dim=-1
                                ).tolist()
                                exp_update_padded, exp_update_pad = pad_dataproto_to_divisor(
                                    exp_output, self.exp_learner_wg.world_size
                                )
                                actor_output = self.exp_learner_wg.update_actor(exp_update_padded)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update({f"exp_learner/{k_m}": v for k_m, v in actor_output_metrics.items()})

                                all_exp_lens = [
                                    len(self.tokenizer.encode(e, add_special_tokens=False))
                                    for e in experiences
                                ]
                                metrics["reward/mean_correct"] = reward_tensor.mean().item()
                                if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                                    metrics["reward/cross_instance_mean"] = reward_tensor.mean().item()
                                metrics["reward/group_std_mean"] = group_std.mean().item()
                                metrics["experience/length_tokens"] = sum(all_exp_lens) / max(len(all_exp_lens), 1)

                            # ── Roll per-src state forward (index 0 of each src group) ──
                            if self.l2c_mode.reward_scope is L2CRewardScope.SAME_INSTANCE:
                                EXP_PER_SRC = [experiences[src * n] for src in range(B)]
                                LATEST_PER_SRC = [resolve_responses_text[src * n] for src in range(B)]

                            # ── Per-iter dump (if enabled) ──
                            step_dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}")
                            if step_dump_dir is not None:
                                try:
                                    rm = exp_batch.non_tensor_batch.get("reward_model", None)
                                    ds = exp_batch.non_tensor_batch.get("data_source", None)
                                    phase_b_gen_full_prompts = self._decode_prompts_batch(exp_output.batch["prompts"])
                                    if iter_k == 0:
                                        phase_a_records = []
                                        for i in range(B):
                                            gt = rm[i].get("ground_truth") if rm is not None else None
                                            phase_a_records.append({
                                                "problem_idx": i,
                                                "iter_k": 0,
                                                "prompt_text": original_prompts[i],
                                                "full_prompt": phase_a_full_prompts[i],
                                                "response_text": phase_a_responses_text[i],
                                                "ground_truth": gt,
                                                "data_source": str(ds[i]) if ds is not None else None,
                                                "correct": bool(train_rewards[i] == 1.0),
                                            })
                                        self._dump_jsonl(os.path.join(step_dump_dir, "phase_a.jsonl"), phase_a_records)

                                    phase_b_gen_records = []
                                    for i in range(B * n):
                                        src_idx = i // n
                                        phase_b_record = {
                                            "candidate_idx": i,
                                            "source_idx": src_idx,
                                            "iter_k": iter_k,
                                            "exp_learner_input": exp_learner_prompts[src_idx],
                                            "full_prompt": phase_b_gen_full_prompts[i],
                                            "raw_output": raw_exp_texts[i],
                                            "parsed_exp": parsed_exps[i],
                                            "parse_ok": parse_oks[i],
                                            "combined_truncated": experiences[i],
                                        }
                                        if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                                            phase_b_record["reward_cross_mean"] = float(
                                                reward_tensor[i].item()
                                            )
                                            phase_b_record["n_probes"] = len(probe_batch)
                                        phase_b_gen_records.append(phase_b_record)
                                    self._dump_jsonl(os.path.join(step_dump_dir, "phase_b_exp_gen.jsonl"), phase_b_gen_records)

                                    if self.l2c_mode.reward_scope is L2CRewardScope.CROSS_INSTANCE:
                                        probe_prompts = list(
                                            probe_batch.non_tensor_batch["raw_prompt"]
                                        )
                                        probe_rm = probe_batch.non_tensor_batch.get(
                                            "reward_model", None
                                        )
                                        probe_ds = probe_batch.non_tensor_batch.get(
                                            "data_source", None
                                        )
                                        probe_count = len(probe_batch)
                                        phase_c_records = []
                                        for candidate_idx in range(B * n):
                                            for probe_idx in range(probe_count):
                                                flat_idx = candidate_idx * probe_count + probe_idx
                                                reward = float(
                                                    cross_pair_rewards[
                                                        candidate_idx, probe_idx
                                                    ].item()
                                                )
                                                phase_c_records.append(
                                                    {
                                                        "candidate_idx": candidate_idx,
                                                        "source_idx": candidate_idx // n,
                                                        "rollout_idx": candidate_idx % n,
                                                        "probe_idx": probe_idx,
                                                        "iter_k": iter_k,
                                                        "experience_used": experiences[
                                                            candidate_idx
                                                        ],
                                                        "probe_text": probe_prompts[
                                                            probe_idx
                                                        ][-1]["content"],
                                                        "full_prompt": resolve_full_prompts[
                                                            flat_idx
                                                        ],
                                                        "response_text": resolve_responses_text[
                                                            flat_idx
                                                        ],
                                                        "reward": reward,
                                                        "correct": bool(reward),
                                                        "ground_truth": (
                                                            probe_rm[probe_idx].get(
                                                                "ground_truth"
                                                            )
                                                            if probe_rm is not None
                                                            else None
                                                        ),
                                                        "data_source": (
                                                            str(probe_ds[probe_idx])
                                                            if probe_ds is not None
                                                            else None
                                                        ),
                                                    }
                                                )
                                        self._dump_jsonl(
                                            os.path.join(
                                                step_dump_dir,
                                                "phase_c_cross_math.jsonl",
                                            ),
                                            phase_c_records,
                                        )
                                    else:
                                        resolve_rewards_list = reward_tensor.cpu().tolist()
                                        phase_b_resolve_records = []
                                        for i in range(B * n):
                                            src_idx = i // n
                                            gt = rm[src_idx].get("ground_truth") if rm is not None else None
                                            phase_b_resolve_records.append({
                                                "candidate_idx": i,
                                                "source_idx": src_idx,
                                                "iter_k": iter_k,
                                                "experience_used": experiences[i],
                                                "problem_text": original_prompts[src_idx],
                                                "full_prompt": resolve_full_prompts[i],
                                                "response_text": resolve_responses_text[i],
                                                "reward": float(resolve_rewards_list[i]),
                                                "ground_truth": gt,
                                            })
                                        self._dump_jsonl(os.path.join(step_dump_dir, "phase_b_resolve.jsonl"), phase_b_resolve_records)
                                except Exception as e:
                                    print(f"[L2L Dump] step {self.global_steps} iter_k {iter_k} dump failed: {e}")

                        # ── Per-iter log + save + tick ──
                        metrics.update({
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                            "training/iter_k": iter_k,
                        })
                        metrics.update({f"timing_s/{k_t}": v for k_t, v in timing_raw.items()})
                        logger.log(data=metrics, step=self.global_steps)

                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            self._save_checkpoint()

                        progress_bar.update(1)
                        self.global_steps += 1

                        if is_last_step:
                            pprint(f"[L2L] Training complete at step {self.global_steps - 1}")
                            progress_bar.close()
                            print("[L2L] training complete, os.sync + os._exit(0) to bypass Ray shutdown hang", flush=True)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os.sync()
                            os._exit(0)

                else:
                    raise ValueError(f"Unknown trainer stage: {self.config.trainer.stage}")
