# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""L2C mode resolution and cross-instance reward/evaluation helpers.

The same-instance training loops remain in :mod:`ray_trainer` because they are
tightly coupled to its PPO step.  This module owns the orthogonal L2C mode
configuration plus cross-instance behavior, so selecting a different reward
scope no longer requires maintaining a separate trainer branch.
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.l2c_mode import L2CRewardScope
from verl.trainer.ppo.reward import compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask


class L2CCrossInstanceMixin:
    """Cross-instance components mixed into ``RayPPOTrainer``."""

    def _l2c_cross_eval_sizes(self) -> tuple[int, int]:
        source_size = self.config.trainer.get("cross_eval_source_size", None)
        probe_size = self.config.trainer.get("cross_eval_probe_size", None)
        if source_size is None:
            source_size = self.config.trainer.get("held_out_size", 64)
        if probe_size is None:
            probe_size = self.config.trainer.get("probe_size", 64)
        source_size = int(source_size)
        probe_size = int(probe_size)
        if source_size < 1 or probe_size < 1:
            raise ValueError(
                f"cross-instance eval sizes must be positive, got S={source_size}, P={probe_size}"
            )
        return source_size, probe_size

    def _compute_cross_instance_reward_math(
        self,
        experiences,
        probe_batch,
        return_dump_info=False,
    ):
        """Score every candidate on one shared, disjoint math probe pool.

        Flat generation order is candidate-major/probe-minor, so reshaping to
        ``(num_candidates, num_probes)`` preserves candidate identity.
        """

        num_candidates = len(experiences)
        num_probes = len(probe_batch)
        if num_candidates < 1 or num_probes < 1:
            raise ValueError(
                f"cross-instance math reward needs candidates and probes; "
                f"got {num_candidates=} {num_probes=}"
            )

        probe_raw_prompts = list(probe_batch.non_tensor_batch["raw_prompt"])
        tokenized = []
        for experience in experiences:
            for probe_idx in range(num_probes):
                messages = deepcopy(probe_raw_prompts[probe_idx])
                problem = messages[-1]["content"]
                if experience and experience != "No previous experience.":
                    problem = self.experience_solve_prompt_template.format(
                        experience=experience,
                        prompt=problem,
                    )
                messages[-1]["content"] = problem
                tokenized.append(self.train_dataset.re_tokenize(messages))

        gen_batch = DataProto.from_single_dict(
            {
                "input_ids": torch.stack([item["input_ids"] for item in tokenized]),
                "attention_mask": torch.stack([item["attention_mask"] for item in tokenized]),
                "position_ids": torch.stack([item["position_ids"] for item in tokenized]),
            }
        )
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
        }

        world_size = self.actor_rollout_wg.world_size
        configured_chunk = self.config.trainer.get("probe_reward_chunk_size", None)
        if configured_chunk is not None and int(configured_chunk) < 1:
            raise ValueError(
                "trainer.probe_reward_chunk_size must be positive when set"
            )
        total = num_candidates * num_probes
        if configured_chunk is None or total <= int(configured_chunk):
            padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
            padded.meta_info["n"] = 1
            output_padded = self.actor_rollout_wg.generate_sequences(padded)
            output = unpad_dataproto(output_padded, pad_size)
        else:
            chunk_size = max(world_size, (int(configured_chunk) // world_size) * world_size)
            pieces = []
            for chunk_idx, start in enumerate(range(0, total, chunk_size)):
                end = min(start + chunk_size, total)
                chunk = gen_batch.slice(start, end)
                padded, pad_size = pad_dataproto_to_divisor(chunk, world_size)
                padded.meta_info["n"] = 1
                output_padded = self.actor_rollout_wg.generate_sequences(padded)
                pieces.append(unpad_dataproto(output_padded, pad_size))
                print(
                    f"[L2C Cross Reward] math chunk {chunk_idx + 1}/"
                    f"{(total + chunk_size - 1) // chunk_size} ({end - start} prompts)"
                )
            output = DataProto.concat(pieces)

        probe_indices = np.arange(total) % num_probes
        for key in ("reward_model", "data_source"):
            if key in probe_batch.non_tensor_batch:
                output.non_tensor_batch[key] = probe_batch.non_tensor_batch[key][probe_indices]

        reward_tensor, _ = compute_reward(output, self.reward_fn)
        raw = reward_tensor.sum(-1).float()
        pair_rewards = (raw == 1.0).float().view(num_candidates, num_probes)
        candidate_rewards = pair_rewards.mean(dim=-1)
        print(
            f"[L2C Cross Reward] math candidates={num_candidates} probes={num_probes} "
            f"accuracy={candidate_rewards.mean().item():.4f}"
        )

        if not return_dump_info:
            return candidate_rewards
        responses = [
            self.tokenizer.decode(output.batch["responses"][i], skip_special_tokens=True)
            for i in range(total)
        ]
        full_prompts = self._decode_prompts_batch(output.batch["prompts"])
        return candidate_rewards, pair_rewards, responses, full_prompts

    def _l2c_generate_math_phase_a(self, batch, n):
        """Generate and score a math Phase A without mutating ``batch``."""

        gen_batch = DataProto.from_single_dict(
            {
                "input_ids": batch.batch["input_ids"],
                "attention_mask": batch.batch["attention_mask"],
                "position_ids": batch.batch["position_ids"],
            }
        )
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
        }
        padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        padded.meta_info["n"] = n
        output_padded = self.actor_rollout_wg.generate_sequences(padded)
        output = unpad_dataproto(output_padded, pad_size * n)

        source_indices = np.arange(len(batch) * n) // n
        for key in ("reward_model", "data_source"):
            if key in batch.non_tensor_batch:
                output.non_tensor_batch[key] = batch.non_tensor_batch[key][source_indices]
        reward_tensor, _ = compute_reward(output, self.reward_fn)
        accuracy = (reward_tensor.sum(-1) == 1.0).float()
        responses = [
            self.tokenizer.decode(output.batch["responses"][i], skip_special_tokens=True)
            for i in range(len(batch) * n)
        ]
        return output, responses, accuracy

    def _run_l2c_cross_instance_eval_math(self, logger):
        """Evaluate source-derived coaching notes on disjoint math probes."""

        n = int(self.config.actor_rollout_ref.rollout.n)
        source_size, probe_size = self._l2c_cross_eval_sizes()

        val_iter = iter(self.val_dataloader)
        batches = []
        accumulated = 0
        required = source_size + probe_size
        while accumulated < required:
            try:
                batch_dict = next(val_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"val dataloader has only {accumulated} rows; cross-instance eval needs "
                    f"S+P={required}"
                ) from exc
            batch = DataProto.from_single_dict(batch_dict)
            batches.append(batch)
            accumulated += len(batch)
        combined = DataProto.concat(batches)[:required]
        source_batch = combined[:source_size]
        probe_batch = combined[source_size:required]
        source_prompts = list(source_batch.non_tensor_batch["raw_prompt"])
        probe_prompts = list(probe_batch.non_tensor_batch["raw_prompt"])

        dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}_eval")
        paths = {}
        for name in (
            "no_exp_source",
            "no_exp_probe",
            "exp_gen",
            "with_exp_meta",
            "with_exp_vanilla",
        ):
            paths[name] = os.path.join(dump_dir, f"{name}.jsonl") if dump_dir else None
            if paths[name] and os.path.exists(paths[name]):
                os.remove(paths[name])

        cache_path = OmegaConf.select(self.config.trainer, "phase_a_cache_path", default=None)
        cache = {}
        if cache_path:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                for line in cache_file:
                    record = json.loads(line)
                    cache.setdefault(int(record["global_problem_idx"]), []).append(record)
            for records in cache.values():
                records.sort(key=lambda row: int(row["rollout_idx"]))

        if cache:
            source_records = []
            for source_idx in range(source_size):
                records = cache.get(source_idx, [])
                if len(records) < n:
                    raise ValueError(
                        f"Phase A cache needs {n} rollouts for source {source_idx}; "
                        f"found {len(records)}"
                    )
                source_records.extend(records[:n])
            source_responses = [row["response_text"] for row in source_records]
            source_accuracy = torch.tensor([float(row["correct"]) for row in source_records])
            if paths["no_exp_source"]:
                self._dump_jsonl(paths["no_exp_source"], source_records)
        else:
            source_output, source_responses, source_accuracy = self._l2c_generate_math_phase_a(
                source_batch, n
            )
            if paths["no_exp_source"]:
                full_prompts = self._decode_prompts_batch(source_output.batch["prompts"])
                reward_model = source_batch.non_tensor_batch.get("reward_model", None)
                data_source = source_batch.non_tensor_batch.get("data_source", None)
                records = []
                for flat_idx in range(source_size * n):
                    source_idx = flat_idx // n
                    records.append(
                        {
                            "tag": "source",
                            "global_problem_idx": source_idx,
                            "rollout_idx": flat_idx % n,
                            "prompt_text": source_prompts[source_idx][-1]["content"],
                            "full_prompt": full_prompts[flat_idx],
                            "response_text": source_responses[flat_idx],
                            "ground_truth": (
                                reward_model[source_idx].get("ground_truth")
                                if reward_model is not None
                                else None
                            ),
                            "data_source": (
                                str(data_source[source_idx]) if data_source is not None else None
                            ),
                            "correct": bool(source_accuracy[flat_idx].item()),
                        }
                    )
                self._dump_jsonl(paths["no_exp_source"], records)

        probe_output, _, probe_accuracy = self._l2c_generate_math_phase_a(probe_batch, n)
        if paths["no_exp_probe"]:
            full_prompts = self._decode_prompts_batch(probe_output.batch["prompts"])
            probe_responses = [
                self.tokenizer.decode(
                    probe_output.batch["responses"][i], skip_special_tokens=True
                )
                for i in range(probe_size * n)
            ]
            reward_model = probe_batch.non_tensor_batch.get("reward_model", None)
            data_source = probe_batch.non_tensor_batch.get("data_source", None)
            records = []
            for flat_idx in range(probe_size * n):
                probe_idx = flat_idx // n
                records.append(
                    {
                        "tag": "probe",
                        "global_problem_idx": source_size + probe_idx,
                        "rollout_idx": flat_idx % n,
                        "prompt_text": probe_prompts[probe_idx][-1]["content"],
                        "full_prompt": full_prompts[flat_idx],
                        "response_text": probe_responses[flat_idx],
                        "ground_truth": (
                            reward_model[probe_idx].get("ground_truth")
                            if reward_model is not None
                            else None
                        ),
                        "data_source": str(data_source[probe_idx]) if data_source is not None else None,
                        "correct": bool(probe_accuracy[flat_idx].item()),
                    }
                )
            self._dump_jsonl(paths["no_exp_probe"], records)

        exp_prompts = []
        for flat_idx in range(source_size * n):
            source_idx = flat_idx // n
            latest = (
                f"Input: {source_prompts[source_idx][-1]['content']}\n"
                f"Output: {source_responses[flat_idx]}"
            )
            exp_prompts.append(
                self.experience_update_prompt.format(
                    PREVIOUS_EXPERIENCE="No previous experience.",
                    LATEST_EXPERIENCE=latest,
                )
            )
        tokenized = [
            self.train_dataset.re_tokenize([{"role": "user", "content": prompt}])
            for prompt in exp_prompts
        ]
        exp_batch = DataProto.from_single_dict(
            {
                "input_ids": torch.stack([item["input_ids"] for item in tokenized]),
                "attention_mask": torch.stack([item["attention_mask"] for item in tokenized]),
                "position_ids": torch.stack([item["position_ids"] for item in tokenized]),
            }
        )
        exp_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
            "n": 1,
        }
        padded, pad_size = pad_dataproto_to_divisor(exp_batch, self.exp_learner_wg.world_size)
        exp_output_padded = self.exp_learner_wg.generate_sequences(padded)
        exp_output = unpad_dataproto(exp_output_padded, pad_size)

        experiences = []
        raw_outputs = []
        parsed_outputs = []
        parse_ok = []
        for flat_idx in range(source_size * n):
            raw = self.tokenizer.decode(
                exp_output.batch["responses"][flat_idx], skip_special_tokens=True
            )
            parsed = self._parse_experience(raw)
            experience = parsed if parsed else "No previous experience."
            experience = self._truncate_experience(
                experience, self.config.trainer.experience_max_length
            )
            raw_outputs.append(raw)
            parsed_outputs.append(parsed)
            parse_ok.append(bool(parsed))
            experiences.append(experience)

        if paths["exp_gen"]:
            full_prompts = self._decode_prompts_batch(exp_output.batch["prompts"])
            records = []
            for flat_idx in range(source_size * n):
                records.append(
                    {
                        "candidate_idx": flat_idx,
                        "source_idx": flat_idx // n,
                        "rollout_idx": flat_idx % n,
                        "exp_learner_input": exp_prompts[flat_idx],
                        "full_prompt": full_prompts[flat_idx],
                        "raw_output": raw_outputs[flat_idx],
                        "parsed_exp": parsed_outputs[flat_idx],
                        "parse_ok": parse_ok[flat_idx],
                        "experience_used": experiences[flat_idx],
                    }
                )
            self._dump_jsonl(paths["exp_gen"], records)

        if dump_dir:
            cross_rewards, pair_rewards, cross_responses, cross_prompts = (
                self._compute_cross_instance_reward_math(
                    experiences,
                    probe_batch,
                    return_dump_info=True,
                )
            )
        else:
            cross_rewards = self._compute_cross_instance_reward_math(
                experiences,
                probe_batch,
            )
        if paths["with_exp_meta"]:
            reward_model = probe_batch.non_tensor_batch.get("reward_model", None)
            data_source = probe_batch.non_tensor_batch.get("data_source", None)
            records = []
            for candidate_idx in range(source_size * n):
                for probe_idx in range(probe_size):
                    flat_idx = candidate_idx * probe_size + probe_idx
                    reward = float(pair_rewards[candidate_idx, probe_idx].item())
                    records.append(
                        {
                            "candidate_idx": candidate_idx,
                            "source_idx": candidate_idx // n,
                            "rollout_idx": candidate_idx % n,
                            "probe_idx": probe_idx,
                            "experience_used": experiences[candidate_idx],
                            "probe_prompt_text": probe_prompts[probe_idx][-1]["content"],
                            "full_prompt": cross_prompts[flat_idx],
                            "response_text": cross_responses[flat_idx],
                            "reward": reward,
                            "correct": bool(reward),
                            "ground_truth": (
                                reward_model[probe_idx].get("ground_truth")
                                if reward_model is not None
                                else None
                            ),
                            "data_source": (
                                str(data_source[probe_idx]) if data_source is not None else None
                            ),
                        }
                    )
            self._dump_jsonl(paths["with_exp_meta"], records)

        if dump_dir:
            same_rewards, same_responses, same_prompts, _ = (
                self._compute_binary_reward_math(
                    experiences,
                    source_batch,
                    source_prompts,
                    n,
                    return_dump_info=True,
                )
            )
        else:
            same_rewards = self._compute_binary_reward_math(
                experiences,
                source_batch,
                source_prompts,
                n,
            )
        if paths["with_exp_vanilla"]:
            reward_model = source_batch.non_tensor_batch.get("reward_model", None)
            data_source = source_batch.non_tensor_batch.get("data_source", None)
            records = []
            for candidate_idx in range(source_size * n):
                source_idx = candidate_idx // n
                reward = float(same_rewards[candidate_idx].item())
                records.append(
                    {
                        "candidate_idx": candidate_idx,
                        "source_idx": source_idx,
                        "rollout_idx": candidate_idx % n,
                        "experience_used": experiences[candidate_idx],
                        "source_prompt_text": source_prompts[source_idx][-1]["content"],
                        "full_prompt": same_prompts[candidate_idx],
                        "response_text": same_responses[candidate_idx],
                        "reward": reward,
                        "correct": bool(reward),
                        "ground_truth": (
                            reward_model[source_idx].get("ground_truth")
                            if reward_model is not None
                            else None
                        ),
                        "data_source": (
                            str(data_source[source_idx]) if data_source is not None else None
                        ),
                    }
                )
            self._dump_jsonl(paths["with_exp_vanilla"], records)

        no_exp_source = source_accuracy.mean().item()
        no_exp_probe = probe_accuracy.mean().item()
        cross_accuracy = cross_rewards.mean().item()
        same_accuracy = same_rewards.mean().item()
        metrics = {
            "eval/acc_no_exp_source": no_exp_source,
            "eval/acc_no_exp_probe": no_exp_probe,
            "eval/acc_cross_instance": cross_accuracy,
            "eval/delta_cross_instance": cross_accuracy - no_exp_probe,
            "eval/acc_same_instance": same_accuracy,
            "eval/delta_same_instance": same_accuracy - no_exp_source,
            # Historical aliases retained for existing notebooks.
            "eval/acc_meta_mean": cross_accuracy,
            "eval/acc_meta_delta": cross_accuracy - no_exp_probe,
            "eval/acc_vanilla_mean": same_accuracy,
            "eval/acc_vanilla_delta": same_accuracy - no_exp_source,
            "eval/parse_success_rate": sum(parse_ok) / max(len(parse_ok), 1),
            "eval/n_sources": source_size,
            "eval/n_probes": probe_size,
            "eval/n_candidates": source_size * n,
            "eval/global_step": self.global_steps,
        }
        print(f"[L2C Cross Eval Math] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if dump_dir:
            summary_path = os.path.join(dump_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(
                    {
                        "experiment_name": self.config.trainer.experiment_name,
                        "reward_scope": L2CRewardScope.CROSS_INSTANCE.value,
                        "global_step": self.global_steps,
                        "n": n,
                        "S": source_size,
                        "P": probe_size,
                        **metrics,
                    },
                    summary_file,
                    ensure_ascii=False,
                    indent=2,
                )
                summary_file.flush()
                os.fsync(summary_file.fileno())

    @staticmethod
    def _l2c_strip_env_prologue(observation, step_num):
        """Remove rule text that must not leak into the coach trajectory."""

        if step_num != 0:
            return observation
        if "Sokoban" in observation:
            return observation.replace(
                "You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets.\n        When you are right next to a box, you can push it by moving in the same direction.\n        You cannot push a box through a wall, and you cannot pull a box.\n        On the board, objects are represented as: \n        - The player (you) appears as 'P' \n        - Walls are represented with '#' \n        - Boxes are marked as 'X' \n        - Empty goals are shown with a 'O'\n        - Boxes on goals are visualized with '√'\n        You can also use [w] for up, [a] for left, [s] for down, and [d] for right.",
                "",
            ).strip()
        if "Frozen Lake" in observation:
            return observation.replace(
                "Welcome to Frozen Lake!\n\nYou are represented by 'P' on the grid.\nGrid symbols:\n  ' ' = Frozen surface (safe to walk on)\n  'H' = Hole (fall in and lose!)\n  'G' = Goal (reach this to win!)\n  'P' = Your current position\n\nAvailable actions: up, down, left, right (or w, a, s, d)\nType your action as: [up], [down], [left], [right] or [w], [a], [s], [d]\n\nObjective: Navigate from the start (top-left) to the goal (bottom-right) without falling into any holes!\n\n",
                "",
            ).strip()
        return observation

    def _l2c_textgame_history(self, trajectory):
        history_text = ""
        for step in trajectory.get("history", []):
            step_num = step["step"]
            observation = self._l2c_strip_env_prologue(
                step.get("current_step_observation", ""), step_num
            )
            response = step.get("raw_response", "")
            history_text += (
                f"\nRound{step_num}_Input: {observation}"
                f"\n\nRound{step_num}_Output: {response}"
            )
        if bool(self.config.trainer.textgame_wfeedback):
            history_text += f"\n\n\n{trajectory.get('final_feedback', '')}\n"
        return history_text

    def _l2c_generate_textgame_batch(self, experiences, seeds, num_steps, validate):
        """Run an exact-size textgame batch, padding only worker transport.

        ``generate_sequences_textgame`` requires ``env_num`` to divide the
        actor world size.  Padding here keeps paper-level pool sizes such as
        250 probes exact instead of silently rounding them down.
        """

        if len(experiences) != len(seeds):
            raise ValueError(
                f"textgame experiences/seeds mismatch: {len(experiences)} != {len(seeds)}"
            )
        requested = len(seeds)
        if requested < 1:
            raise ValueError("textgame generation requires at least one environment")
        world_size = self.actor_rollout_wg.world_size
        pad_size = (-requested) % world_size
        padded_experiences = list(experiences) + [""] * pad_size
        padded_seeds = list(seeds)
        used_seeds = {int(seed) for seed in seeds}
        dummy_seed = 2_000_000_000
        for _ in range(pad_size):
            while dummy_seed in used_seeds:
                dummy_seed -= 1
            if dummy_seed < 0:
                raise ValueError("could not allocate a disjoint textgame transport seed")
            padded_seeds.append(dummy_seed)
            used_seeds.add(dummy_seed)
            dummy_seed -= 1
        output = self.actor_rollout_wg.generate_sequences_textgame(
            env_config=self.textgame_env_config,
            env_num=requested + pad_size,
            tokenizer=self.tokenizer,
            experiences=padded_experiences,
            num_steps=num_steps,
            seeds=padded_seeds,
            validate=validate,
        )
        if isinstance(output, list):
            output = output[0]
        trajectories = [output["env_trajectories"].get(idx, {}) for idx in range(requested)]
        rewards = [float(output["reward_list"][idx][0]) for idx in range(requested)]
        return trajectories, rewards

    def _l2c_tokenize_textgame_prompts(self, prompts):
        """Apply the coach chat template and enforce left padding for vLLM."""

        enable_thinking = not self.textgame_env_config["no_think"]
        max_prompt_length = int(self.config.data["max_prompt_length"])
        input_ids = []
        attention_masks = []
        position_ids = []
        for prompt in prompts:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            tokenized = self.tokenizer(
                chat,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ids_2d, mask_2d = verl_F.postprocess_data(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )
            input_ids.append(ids_2d[0])
            attention_masks.append(mask_2d[0])
            position_ids.append(compute_position_id_with_mask(mask_2d[0]))
        return DataProto.from_single_dict(
            {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_masks),
                "position_ids": torch.stack(position_ids),
            }
        )

    def _run_l2c_cross_instance_train_textgame_step(
        self,
        n,
        metrics,
        timing_raw,
    ):
        """Run one textgame step with a disjoint probe pool per source group.

        All n candidates from one source share that group's probe_size seeds.
        Different groups and training steps use different source/probe seeds.
        """

        source_size = int(self.config.data.train_batch_size)
        probe_size = int(self.config.trainer.get("probe_size", 8))
        if source_size < 1 or probe_size < 1:
            raise ValueError(f"cross-instance textgame training needs positive G/P, got {source_size}/{probe_size}")
        num_steps = int(self.config.trainer.textgame_max_steps)
        dump_enabled = bool(self.config.trainer.get("dump_dir", None))

        oel_round = int(self.config.trainer.get("oel_round", 0) or 0)
        group_size = 1 + probe_size
        seeds_per_step = source_size * group_size
        seed_start = self.global_steps * seeds_per_step
        seed_base = 505019424 + 90039 + 100000 + oel_round * 10000000
        step_seeds = [seed_base + (seed_start + idx) * 1000 for idx in range(seeds_per_step)]
        # Each group owns one source followed by probe_size probes, matching
        # guanheng/l2l-meta. Consecutive step blocks never reuse logical seeds.
        source_seeds = step_seeds[::group_size]
        probe_seeds = [
            step_seeds[source_idx * group_size + 1 + probe_idx]
            for source_idx in range(source_size)
            for probe_idx in range(probe_size)
        ]

        with marked_timer("step", timing_raw):
            with marked_timer("phase_a", timing_raw, color="blue"):
                source_trajectories, source_rewards = self._l2c_generate_textgame_batch(
                    [""] * source_size,
                    source_seeds,
                    num_steps,
                    validate=False,
                )
                source_accuracy = (torch.tensor(source_rewards) == 1.0).float()
                metrics["l2l/train_acc"] = source_accuracy.mean().item()

            with marked_timer("phase_b_gen", timing_raw, color="orange"):
                exp_prompts = [
                    self.experience_update_prompt.format(
                        PREVIOUS_EXPERIENCE="No previous experience.",
                        LATEST_EXPERIENCE=self._l2c_textgame_history(trajectory),
                    )
                    for trajectory in source_trajectories
                ]
                debug_idx = random.randint(0, source_size - 1)
                debug_prompt = exp_prompts[debug_idx]
                print(
                    f"[L2L EXP INPUT DEBUG] step={self.global_steps} scope=cross_instance "
                    f"idx={debug_idx} len_chars={len(debug_prompt)}"
                )
                print(f"  head 800: {debug_prompt[:800]!r}")
                print(f"  tail 400: {debug_prompt[-400:]!r}")

                exp_batch = self._l2c_tokenize_textgame_prompts(exp_prompts)
                exp_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": True,
                    "n": n,
                }
                padded, pad_size = pad_dataproto_to_divisor(
                    exp_batch, self.exp_learner_wg.world_size
                )
                exp_output_padded = self.exp_learner_wg.generate_sequences(padded)
                exp_output = unpad_dataproto(exp_output_padded, pad_size * n)

                num_candidates = source_size * n
                experiences = []
                raw_outputs = []
                parsed_outputs = []
                parse_ok = []
                failed_logged = False
                sample_idx = random.randint(0, num_candidates - 1)
                for candidate_idx in range(num_candidates):
                    raw = self.tokenizer.decode(
                        exp_output.batch["responses"][candidate_idx], skip_special_tokens=True
                    )
                    parsed = self._parse_experience(raw)
                    experience = (
                        self._truncate_experience(
                            parsed, self.config.trainer.experience_max_length
                        )
                        if parsed
                        else ""
                    )
                    raw_outputs.append(raw)
                    parsed_outputs.append(parsed)
                    parse_ok.append(bool(parsed))
                    experiences.append(experience)
                    if candidate_idx == sample_idx:
                        print(
                            f"[L2L EXP DEBUG] step={self.global_steps} "
                            f"scope=cross_instance parsed_len={len(parsed)}"
                        )
                        print(f"  raw exp_text head: {raw[:300]!r}")
                        print(f"  raw exp_text tail: {raw[-200:]!r}")
                        print(f"  parsed head: {parsed[:200]!r}")
                    if not parsed and not failed_logged:
                        failed_logged = True
                        print(
                            f"[L2L EXP FAIL DEBUG] step={self.global_steps} "
                            f"scope=cross_instance idx={candidate_idx} exp_text_len={len(raw)}"
                        )
                        print(f"  raw exp_text head: {raw[:500]!r}")
                        print(f"  raw exp_text tail: {raw[-300:]!r}")
                metrics["experience/parse_success_rate"] = sum(parse_ok) / max(num_candidates, 1)
                metrics["experience/n_parse_failed"] = num_candidates - sum(parse_ok)

            with marked_timer("phase_c_cross", timing_raw, color="red"):
                pair_rewards = torch.zeros(num_candidates, probe_size)
                phase_c_trajectories = (
                    [None] * (num_candidates * probe_size)
                    if dump_enabled
                    else None
                )
                # One round per candidate slot, with (source, probe) ordering.
                # Each source's probes receive only that source's candidate.
                for rollout_idx in range(n):
                    round_experiences = [
                        experiences[source_idx * n + rollout_idx]
                        for source_idx in range(source_size)
                        for _ in range(probe_size)
                    ]
                    trajectories, rewards = self._l2c_generate_textgame_batch(
                        round_experiences,
                        probe_seeds,
                        num_steps,
                        validate=False,
                    )
                    for env_idx, reward in enumerate(rewards):
                        source_idx, probe_idx = divmod(env_idx, probe_size)
                        candidate_idx = source_idx * n + rollout_idx
                        pair_rewards[candidate_idx, probe_idx] = float(reward == 1.0)
                        if phase_c_trajectories is not None:
                            phase_c_trajectories[
                                candidate_idx * probe_size + probe_idx
                            ] = trajectories[env_idx]
                candidate_rewards = pair_rewards.mean(dim=-1)
                reward_matrix = candidate_rewards.view(source_size, n)

            with marked_timer("update_exp_learner", timing_raw, color="green"):
                group_mean = reward_matrix.mean(dim=-1, keepdim=True)
                group_std = reward_matrix.std(dim=-1, keepdim=True)
                advantages = ((reward_matrix - group_mean) / (group_std + 1e-8)).reshape(-1)

                prompt_length = exp_batch.batch["input_ids"].shape[-1]
                response_length = exp_output.batch["responses"].shape[-1]
                response_mask = exp_output.batch["attention_mask"][:, prompt_length:]
                response_mask = response_mask[:, :response_length]
                exp_output.batch["advantages"] = (
                    advantages.unsqueeze(-1).to(response_mask.device) * response_mask
                )
                exp_output.meta_info["recompute_log_prob"] = True
                compute_padded, compute_pad = pad_dataproto_to_divisor(
                    exp_output, self.exp_learner_wg.world_size
                )
                old_log_prob_padded = self.exp_learner_wg.compute_log_prob(compute_padded)
                old_log_prob = unpad_dataproto(old_log_prob_padded, compute_pad)
                exp_output.batch["old_log_probs"] = old_log_prob.batch["old_log_probs"]
                exp_output.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                exp_output.meta_info["l2l_mode"] = True
                exp_output.meta_info["global_token_num"] = torch.sum(
                    exp_output.batch["attention_mask"], dim=-1
                ).tolist()
                update_padded, _ = pad_dataproto_to_divisor(
                    exp_output, self.exp_learner_wg.world_size
                )
                update_output = self.exp_learner_wg.update_actor(update_padded)
                update_metrics = reduce_metrics(update_output.meta_info["metrics"])
                metrics.update(
                    {f"exp_learner/{key}": value for key, value in update_metrics.items()}
                )

                experience_lengths = [
                    len(self.tokenizer.encode(experience, add_special_tokens=False))
                    for experience in experiences
                ]
                metrics["reward/mean_correct"] = candidate_rewards.mean().item()
                metrics["reward/cross_instance_mean"] = candidate_rewards.mean().item()
                metrics["reward/group_std_mean"] = group_std.mean().item()
                metrics["experience/length_tokens"] = sum(experience_lengths) / max(
                    len(experience_lengths), 1
                )

            dump_dir = self._l2l_dump_dir(f"step_{self.global_steps:06d}")
            if dump_dir:
                phase_a_records = []
                for source_idx, trajectory in enumerate(source_trajectories):
                    history = trajectory.get("history", [])
                    phase_a_records.append(
                        {
                            "env_idx": source_idx,
                            "seed": source_seeds[source_idx],
                            "reward": source_rewards[source_idx],
                            "correct": bool(source_rewards[source_idx] == 1.0),
                            "n_steps": len(history),
                            "stop_reason": trajectory.get("stop_reason", "unknown"),
                            "history": [
                                {
                                    "step": step["step"],
                                    "observation": step.get("current_step_observation", ""),
                                    "response": step.get("raw_response", ""),
                                }
                                for step in history
                            ],
                            "final_feedback": trajectory.get("final_feedback", ""),
                        }
                    )
                self._dump_jsonl(
                    os.path.join(dump_dir, "phase_a_textgame.jsonl"), phase_a_records
                )

                phase_b_records = []
                for candidate_idx in range(num_candidates):
                    source_idx = candidate_idx // n
                    rollout_idx = candidate_idx % n
                    phase_b_records.append(
                        {
                            "candidate_idx": candidate_idx,
                            "env_idx": source_idx,
                            "rollout_idx": rollout_idx,
                            "source_seed": source_seeds[source_idx],
                            "probe_seeds": probe_seeds[
                                source_idx * probe_size:(source_idx + 1) * probe_size
                            ],
                            "exp_learner_input": exp_prompts[source_idx],
                            "raw_output": raw_outputs[candidate_idx],
                            "parsed_exp": parsed_outputs[candidate_idx],
                            "parse_ok": parse_ok[candidate_idx],
                            "experience_used": experiences[candidate_idx],
                            "reward_cross_mean": float(candidate_rewards[candidate_idx].item()),
                        }
                    )
                self._dump_jsonl(
                    os.path.join(dump_dir, "phase_b_exp_gen.jsonl"), phase_b_records
                )

                phase_c_records = []
                for candidate_idx in range(num_candidates):
                    source_idx = candidate_idx // n
                    group_probe_seeds = probe_seeds[
                        source_idx * probe_size:(source_idx + 1) * probe_size
                    ]
                    for probe_idx, probe_seed in enumerate(group_probe_seeds):
                        trajectory = phase_c_trajectories[
                            candidate_idx * probe_size + probe_idx
                        ] or {}
                        history = trajectory.get("history", [])
                        reward = float(pair_rewards[candidate_idx, probe_idx].item())
                        phase_c_records.append(
                            {
                                "candidate_idx": candidate_idx,
                                "env_idx": candidate_idx // n,
                                "rollout_idx": candidate_idx % n,
                                "probe_idx": probe_idx,
                                "seed": probe_seed,
                                "experience_used": experiences[candidate_idx],
                                "reward": reward,
                                "correct": bool(reward),
                                "n_steps": len(history),
                                "stop_reason": trajectory.get("stop_reason", "unknown"),
                                "history": [
                                    {
                                        "step": step["step"],
                                        "observation": step.get(
                                            "current_step_observation", ""
                                        ),
                                        "response": step.get("raw_response", ""),
                                    }
                                    for step in history
                                ],
                            }
                        )
                self._dump_jsonl(
                    os.path.join(dump_dir, "phase_c_cross_textgame.jsonl"), phase_c_records
                )

    def _run_l2c_cross_instance_eval_textgame(self, logger):
        """Evaluate source-seed coaching notes on a disjoint probe-seed pool."""

        n = int(self.config.actor_rollout_ref.rollout.n)
        source_size, probe_size = self._l2c_cross_eval_sizes()
        num_steps = int(self.config.trainer.textgame_max_steps)
        seed_base = 468382021 + 78025
        source_seeds = [seed_base + idx * 1000 for idx in range(source_size)]
        probe_start_idx = max(500, source_size)
        probe_seeds = [
            seed_base + (probe_start_idx + idx) * 1000 for idx in range(probe_size)
        ]

        dump_dir = self._l2l_dump_dir(
            f"step_{self.global_steps:06d}_eval_textgame"
        )
        paths = {}
        for name in (
            "no_exp_source",
            "no_exp_probe",
            "exp_gen",
            "with_exp_meta",
            "with_exp_vanilla",
        ):
            paths[name] = os.path.join(dump_dir, f"{name}.jsonl") if dump_dir else None
            if paths[name] and os.path.exists(paths[name]):
                os.remove(paths[name])

        source_trajectories = [None] * (source_size * n)
        source_rewards = [0.0] * (source_size * n)
        cache_path = OmegaConf.select(self.config.trainer, "phase_a_cache_path", default=None)
        if cache_path:
            cache = {}
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                for line in cache_file:
                    record = json.loads(line)
                    cache[(int(record["env_idx"]), int(record["rollout_idx"]))] = record
            for rollout_idx in range(n):
                for source_idx in range(source_size):
                    record = cache.get((source_idx, rollout_idx))
                    if record is None:
                        raise ValueError(
                            f"Phase A textgame cache missing source={source_idx}, rollout={rollout_idx}"
                        )
                    if int(record["seed"]) != source_seeds[source_idx]:
                        raise ValueError(
                            f"Phase A textgame cache seed mismatch for source {source_idx}: "
                            f"{record['seed']} != {source_seeds[source_idx]}"
                        )
                    flat_idx = rollout_idx * source_size + source_idx
                    source_trajectories[flat_idx] = {
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
                    source_rewards[flat_idx] = float(record["reward"])
        else:
            for rollout_idx in range(n):
                trajectories, rewards = self._l2c_generate_textgame_batch(
                    [""] * source_size,
                    source_seeds,
                    num_steps,
                    validate=True,
                )
                for source_idx in range(source_size):
                    flat_idx = rollout_idx * source_size + source_idx
                    source_trajectories[flat_idx] = trajectories[source_idx]
                    source_rewards[flat_idx] = rewards[source_idx]

        def _trajectory_record(tag, env_idx, rollout_idx, seed, reward, trajectory):
            history = trajectory.get("history", [])
            return {
                "tag": tag,
                "env_idx": env_idx,
                "rollout_idx": rollout_idx,
                "seed": seed,
                "reward": reward,
                "correct": bool(reward == 1.0),
                "n_steps": len(history),
                "stop_reason": trajectory.get("stop_reason", "unknown"),
                "history": [
                    {
                        "step": step["step"],
                        "observation": step.get("current_step_observation", ""),
                        "response": step.get("raw_response", ""),
                    }
                    for step in history
                ],
                "final_feedback": trajectory.get("final_feedback", ""),
            }

        if paths["no_exp_source"]:
            records = []
            for flat_idx, trajectory in enumerate(source_trajectories):
                source_idx = flat_idx % source_size
                rollout_idx = flat_idx // source_size
                records.append(
                    _trajectory_record(
                        "source",
                        source_idx,
                        rollout_idx,
                        source_seeds[source_idx],
                        source_rewards[flat_idx],
                        trajectory,
                    )
                )
            self._dump_jsonl(paths["no_exp_source"], records)

        probe_rewards = [0.0] * (probe_size * n)
        probe_records = []
        for rollout_idx in range(n):
            trajectories, rewards = self._l2c_generate_textgame_batch(
                [""] * probe_size,
                probe_seeds,
                num_steps,
                validate=True,
            )
            for probe_idx in range(probe_size):
                flat_idx = rollout_idx * probe_size + probe_idx
                probe_rewards[flat_idx] = rewards[probe_idx]
                if paths["no_exp_probe"]:
                    probe_records.append(
                        _trajectory_record(
                            "probe",
                            probe_idx,
                            rollout_idx,
                            probe_seeds[probe_idx],
                            rewards[probe_idx],
                            trajectories[probe_idx],
                        )
                    )
        if paths["no_exp_probe"]:
            self._dump_jsonl(paths["no_exp_probe"], probe_records)

        exp_prompts = [
            self.experience_update_prompt.format(
                PREVIOUS_EXPERIENCE="No previous experience.",
                LATEST_EXPERIENCE=self._l2c_textgame_history(trajectory),
            )
            for trajectory in source_trajectories
        ]
        exp_batch = self._l2c_tokenize_textgame_prompts(exp_prompts)
        exp_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
            "n": 1,
        }
        padded, pad_size = pad_dataproto_to_divisor(exp_batch, self.exp_learner_wg.world_size)
        exp_output_padded = self.exp_learner_wg.generate_sequences(padded)
        exp_output = unpad_dataproto(exp_output_padded, pad_size)

        num_candidates = source_size * n
        experiences = []
        raw_outputs = []
        parsed_outputs = []
        parse_ok = []
        for candidate_idx in range(num_candidates):
            raw = self.tokenizer.decode(
                exp_output.batch["responses"][candidate_idx], skip_special_tokens=True
            )
            parsed = self._parse_experience(raw)
            experience = (
                self._truncate_experience(parsed, self.config.trainer.experience_max_length)
                if parsed
                else ""
            )
            raw_outputs.append(raw)
            parsed_outputs.append(parsed)
            parse_ok.append(bool(parsed))
            experiences.append(experience)

        if paths["exp_gen"]:
            records = []
            for candidate_idx in range(num_candidates):
                source_idx = candidate_idx % source_size
                rollout_idx = candidate_idx // source_size
                records.append(
                    {
                        "candidate_idx": candidate_idx,
                        "source_env_idx": source_idx,
                        "rollout_idx": rollout_idx,
                        "source_seed": source_seeds[source_idx],
                        "exp_learner_input": exp_prompts[candidate_idx],
                        "raw_output": raw_outputs[candidate_idx],
                        "parsed_exp": parsed_outputs[candidate_idx],
                        "parse_ok": parse_ok[candidate_idx],
                        "experience_used": experiences[candidate_idx],
                    }
                )
            self._dump_jsonl(paths["exp_gen"], records)

        pair_rewards = torch.zeros(num_candidates, probe_size)
        for probe_idx, probe_seed in enumerate(probe_seeds):
            trajectories, rewards = self._l2c_generate_textgame_batch(
                experiences,
                [probe_seed] * num_candidates,
                num_steps,
                validate=True,
            )
            records = []
            for candidate_idx, reward in enumerate(rewards):
                pair_rewards[candidate_idx, probe_idx] = float(reward == 1.0)
                if paths["with_exp_meta"]:
                    trajectory = trajectories[candidate_idx]
                    history = trajectory.get("history", [])
                    records.append(
                        {
                            "candidate_idx": candidate_idx,
                            "source_env_idx": candidate_idx % source_size,
                            "rollout_idx": candidate_idx // source_size,
                            "probe_idx": probe_idx,
                            "probe_seed": probe_seed,
                            "experience_used": experiences[candidate_idx],
                            "reward": float(reward),
                            "correct": bool(reward == 1.0),
                            "n_steps": len(history),
                            "stop_reason": trajectory.get("stop_reason", "unknown"),
                            "history": [
                                {
                                    "step": step["step"],
                                    "observation": step.get(
                                        "current_step_observation", ""
                                    ),
                                    "response": step.get("raw_response", ""),
                                }
                                for step in history
                            ],
                        }
                    )
            if paths["with_exp_meta"]:
                self._dump_jsonl(
                    paths["with_exp_meta"], records, append=(probe_idx > 0)
                )

        cross_rewards = pair_rewards.mean(dim=-1)
        same_seeds = [source_seeds[idx % source_size] for idx in range(num_candidates)]
        same_trajectories, same_raw_rewards = self._l2c_generate_textgame_batch(
            experiences,
            same_seeds,
            num_steps,
            validate=True,
        )
        same_rewards = (torch.tensor(same_raw_rewards) == 1.0).float()
        if paths["with_exp_vanilla"]:
            records = []
            for candidate_idx, trajectory in enumerate(same_trajectories):
                history = trajectory.get("history", [])
                records.append(
                    {
                        "candidate_idx": candidate_idx,
                        "source_env_idx": candidate_idx % source_size,
                        "rollout_idx": candidate_idx // source_size,
                        "source_seed": same_seeds[candidate_idx],
                        "experience_used": experiences[candidate_idx],
                        "reward": float(same_rewards[candidate_idx].item()),
                        "correct": bool(same_rewards[candidate_idx].item()),
                        "n_steps": len(history),
                        "stop_reason": trajectory.get("stop_reason", "unknown"),
                        "history": [
                            {
                                "step": step["step"],
                                "observation": step.get("current_step_observation", ""),
                                "response": step.get("raw_response", ""),
                            }
                            for step in history
                        ],
                    }
                )
            self._dump_jsonl(paths["with_exp_vanilla"], records)

        no_exp_source = (torch.tensor(source_rewards) == 1.0).float().mean().item()
        no_exp_probe = (torch.tensor(probe_rewards) == 1.0).float().mean().item()
        cross_accuracy = cross_rewards.mean().item()
        same_accuracy = same_rewards.mean().item()
        metrics = {
            "eval/acc_no_exp_source": no_exp_source,
            "eval/acc_no_exp_probe": no_exp_probe,
            "eval/acc_cross_instance": cross_accuracy,
            "eval/delta_cross_instance": cross_accuracy - no_exp_probe,
            "eval/acc_same_instance": same_accuracy,
            "eval/delta_same_instance": same_accuracy - no_exp_source,
            "eval/acc_meta_mean": cross_accuracy,
            "eval/acc_meta_delta": cross_accuracy - no_exp_probe,
            "eval/acc_vanilla_mean": same_accuracy,
            "eval/acc_vanilla_delta": same_accuracy - no_exp_source,
            "eval/parse_success_rate": sum(parse_ok) / max(len(parse_ok), 1),
            "eval/n_sources": source_size,
            "eval/n_probes": probe_size,
            "eval/n_candidates": num_candidates,
            "eval/global_step": self.global_steps,
        }
        print(f"[L2C Cross Eval Textgame] FINAL @ step {self.global_steps}: {metrics}")
        logger.log(data=metrics, step=self.global_steps)

        if dump_dir:
            summary_path = os.path.join(dump_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(
                    {
                        "experiment_name": self.config.trainer.experiment_name,
                        "reward_scope": L2CRewardScope.CROSS_INSTANCE.value,
                        "global_step": self.global_steps,
                        "n": n,
                        "S": source_size,
                        "P": probe_size,
                        "textgame_env_id": self.config.trainer.textgame_env_id,
                        **metrics,
                    },
                    summary_file,
                    ensure_ascii=False,
                    indent=2,
                )
                summary_file.flush()
                os.fsync(summary_file.fileno())
