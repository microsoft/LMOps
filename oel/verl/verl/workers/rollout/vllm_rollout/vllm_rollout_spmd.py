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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
import textarena as ta
import random
import time

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _extract_action_from_response(response: str, action_dict: dict) -> str:
    """
    Extract action from model response by finding any key from action_dict.
    Returns the value corresponding to the last matching key found in the response.
    
    Args:
        response: The full model response string
        action_dict: Dictionary mapping action keys to action values
    
    Returns:
        Extracted action value corresponding to the last matching key, or 'no valid action' if no match found
    """
    # Find all matches of action_dict keys in the response
    matches = []
    for key in action_dict.keys():
        # Find all occurrences of this key in the response
        for match in re.finditer(re.escape(key), response):
            matches.append((match.start(), key))
    
    if matches:
        # Sort by position and get the last match (highest position)
        matches.sort(key=lambda x: x[0])
        last_match_key = matches[-1][1]
        return action_dict[last_match_key]
    else:
        # If no matches found, return 'no valid action'
        return 'no valid action'


def _extract_minesweeper_action_from_response(response: str) -> str:
    """
    Extract Minesweeper action in the form "[row col]" from model response.
    Always returns the last matched coordinate pair if multiple are present.
    
    Args:
        response: The full model response string
    
    Returns:
        The last matched coordinate pair including brackets (e.g. "[3 4]"),
        or 'no valid action' if no valid coordinate pair is found.
    """
    pattern = r"\[(\d+)\s+(\d+)\]"
    matches = list(re.finditer(pattern, response))
    if matches:
        last_match = matches[-1]
        return last_match.group(0)
    else:
        return "no valid action"


def _format_action(action: str) -> str:
    """
    Format action by adding brackets if missing.
    This replicates the functionality of ActionFormattingWrapper.
    
    Args:
        action: The raw action string
    
    Returns:
        Formatted action with brackets
    """
    if "[" not in action and "]" not in action:
        return f"[{action}]"
    else:
        return action


def _convert_observation_to_string(player_id: int, observation: List, full_observations: Dict, env) -> str:
    """
    Convert observation messages to string format.
    This replicates the functionality of LLMObservationWrapper.
    
    Args:
        player_id: The player ID
        observation: List of (sender_id, message, observation_type) tuples
        full_observations: Dictionary tracking full observations for each player
        env: The TextArena environment instance
    
    Returns:
        Formatted string observation
    """
    import textarena as ta
    # Extend full observations with new messages
    if observation:
        # if observation is string, directly add to full_observations
        if isinstance(observation, str):
            full_observations[player_id].append(("", observation, "text"))
        else:
            full_observations[player_id].extend(observation)
    # Convert to string
    str_observation = ""
    
    for sender_id, message, _ in full_observations[player_id]:
        if sender_id == ta.GAME_ID:
            sender_name = "GAME"
            str_observation += f"\n[{sender_name}] {message}"
        elif sender_id == '':
            str_observation += f"{message}"
        else:
            sender_name = env.state.role_mapping.get(sender_id, f"Player {sender_id}")
            str_observation += f"\n[{sender_name}] {message}"
    
    return str_observation

def _convert_current_step_observation_to_string(player_id: int, observation: List, env) -> str:
    """
    Convert only the current step observation to string format.
    """
    import textarena as ta
    str_observation = ""
    if isinstance(observation, str):
        str_observation = observation
    else:
        for sender_id, message, _ in observation:
            if sender_id == ta.GAME_ID:
                sender_name = "GAME"
                str_observation += f"\n[{sender_name}] {message}"
            else:
                pass
        
    return str_observation


def _make_textgame_envs(env_id: str, num_players: int, env_num: int, size: int, num_holes: int, randomize_start_goal: bool, num_boxes: int, dim_room: tuple, seeds: List[int] = None):
    """Create and reset multiple textgame environments with basic random seeding.
    
    Args:
        env_id: Environment ID
        num_players: Number of players
        env_num: Number of environments to create
        size: Environment size (for FrozenLake)
        num_holes: Number of holes (for FrozenLake)
        randomize_start_goal: Whether to randomize start/goal (for FrozenLake)
        num_boxes: Number of boxes (for Sokoban)
        dim_room: Room dimensions (for Sokoban)
        seeds: Optional list of seeds for each environment. If None, seeds are randomly generated.
               If provided, must have length equal to env_num.
    """
    envs = []
    if seeds is None:
        seeds = [int(time.time() * 1000) % (2**32 - 1) + random.randint(0, 99999) for _ in range(env_num)]
    else:
        assert len(seeds) == env_num, f"Number of seeds ({len(seeds)}) must match env_num ({env_num})"
    
    for i, seed in enumerate(seeds):
        if env_id == "FrozenLake-v0-raw":
            env = ta.make(env_id=env_id, size=size, num_holes=num_holes, randomize_start_goal=randomize_start_goal)
        elif env_id == "Sokoban-v0":
            env = ta.make(env_id=env_id, num_boxes=num_boxes, dim_room=dim_room)
        else:
            env = ta.make(env_id=env_id)
        env.reset(num_players=num_players, seed=seed)
        envs.append(env)
    return envs


def _get_textgame_action_dict(env_id: str):
    """Return the action dictionary or None for coordinate-based games."""
    if "FrozenLake" in env_id:
        action_dict = {
            "[up]": "up",
            "[down]": "down",
            "[left]": "left",
            "[right]": "right",
            "[w]": "up",
            "[a]": "left",
            "[s]": "down",
            "[d]": "right",
        }
    elif "Sokoban" in env_id:
        action_dict = {
            "[up]": "up",
            "[down]": "down",
            "[left]": "left",
            "[right]": "right",
            "[w]": "up",
            "[a]": "left",
            "[s]": "down",
            "[d]": "right",
        }
    elif "Minesweeper" in env_id:
        # Minesweeper uses coordinate-based actions like "[row col]"; no fixed action_dict needed.
        action_dict = None
    else:
        # assert error if env_config is not valid
        raise ValueError(f"Invalid env_config: action extraction is not defined for {env_id}")
    return action_dict


def _build_textgame_prompt(experience: str, observation_str: str) -> str:
    """Construct the natural-language prompt given experience and observation."""
    if "Sokoban" in observation_str:
        observation_no_rules = observation_str.replace("You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets.\n        When you are right next to a box, you can push it by moving in the same direction.\n        You cannot push a box through a wall, and you cannot pull a box.\n        On the board, objects are represented as: \n        - The player (you) appears as 'P' \n        - Walls are represented with '#' \n        - Boxes are marked as 'X' \n        - Empty goals are shown with a 'O'\n        - Boxes on goals are visualized with '√'\n        You can also use [w] for up, [a] for left, [s] for down, and [d] for right.", 
        "You are the player and you are represented by 'P' on the grid. You should select the best action to reach the goal in the shortest number of steps. Your only way to interact is to move one step each time. Available actions: up, down, left, right (or w, a, s, d). Type your action as: [up], [down], [left], [right] or [w], [a], [s], [d]").strip() + "\n"
    elif 'Frozen Lake' in observation_str:
        observation_no_rules = observation_str.replace("Welcome to Frozen Lake!\n\nYou are represented by 'P' on the grid.\nGrid symbols:\n  ' ' = Frozen surface (safe to walk on)\n  'H' = Hole (fall in and lose!)\n  'G' = Goal (reach this to win!)\n  'P' = Your current position\n\nAvailable actions: up, down, left, right (or w, a, s, d)\nType your action as: [up], [down], [left], [right] or [w], [a], [s], [d]\n\nObjective: Navigate from the start (top-left) to the goal (bottom-right) without falling into any holes!\n\n", 
        "You are the player and you are represented by 'P' on the grid. You should select the best action to reach the goal in the shortest number of steps. Your only way to interact is to move one step each time. Available actions: up, down, left, right (or w, a, s, d). Type your action as: [up], [down], [left], [right] or [w], [a], [s], [d]").strip() + "\n"
    else:
        assert False, f"observation_str: {observation_str} is not supported"
    if experience:
        prompt_text = (
            f"You are an agent playing a game on a grid, acting as a reasoning engine.\n\nYour decisions are based on the experience you have learned about the game's rules or strategies. This experience is only a guess of how the game works, and the rules and strategies may be incomplete or incorrect.\n\nGiven experience for rules or strategies you have learned:\n{experience}\n\n"
            f"Current situation:\n{observation_no_rules}\n\n"
            "What action do you take? (Remember to wrap your final answer of the action in square brackets)"
        )
    else:
        prompt_text = (
            f"You are an agent playing a game on a grid, acting as a reasoning engine.\n\n"
            f"Current situation:\n{observation_no_rules}\n\n"
            "What action do you take? (Remember to wrap your final answer of the action in square brackets)"
        )
    return prompt_text


def _tokenize_textgame_prompt(tokenizer, prompt_with_template: str, max_prompt_length: int) -> List[int]:
    """Tokenize prompt and return prompt token ids."""
    prompt_inputs = tokenizer(
        prompt_with_template,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_prompt_length,
    )
    prompt_token_ids = prompt_inputs["input_ids"][0].tolist()
    return prompt_token_ids


def _build_textgame_sampling_kwargs(max_response_length: int, validate: bool = False, val_kwargs=None) -> Dict[str, Any]:
    """Sampling kwargs used for textgame generation."""
    if validate:
        kwargs = {
            "top_k": val_kwargs.top_k,
            "top_p": val_kwargs.top_p,
            "temperature": val_kwargs.temperature,
            "n": 1,
            "max_tokens": max_response_length,
        }
    else:
        kwargs = {
            "max_tokens": max_response_length,
        }
    return kwargs


def _postprocess_textgame_response(raw_response: str, keep_reasoning: bool) -> str:
    """Optionally strip reasoning before </think>, controlled by keep_reasoning flag."""
    if keep_reasoning:
        return raw_response
    if "</think>" in raw_response:
        return raw_response.split("</think>")[-1]
    return "Format error: no </think> found in the response"

class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, role, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        role = kwargs.pop("role", "actor_rollout")
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}
        
        # Disable sleep mode for exp_learner to avoid vLLM sleep mode conflict
        # when both actor_rollout and exp_learner need rollout functionality in the same process
        # enable_sleep_mode = role != "exp_learner"
        enable_sleep_mode = True
        # logger.info(f"vLLM rollout initialized for role '{role}' with enable_sleep_mode={enable_sleep_mode}")
        
        vllm_kwargs = dict(
            model=model_path,
            enable_sleep_mode=enable_sleep_mode,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        class _MyInferenceEngineProxy():

            def __init__(self):
                def _wake_up_func_impl():
                    vllm_kwargs.update({"enable_sleep_mode": False})
                    ret = LLM(**vllm_kwargs)
                    return ret
                self._wake_up_func = _wake_up_func_impl
                self.engine = None
            
            def wake_up(self):
                if self.engine is None:
                    self.engine = self._wake_up_func()
                else:
                    # if the engine is already initialized, we can just return
                    print("vllm inference engine is already initialized, skip wake up")
            
            def sleep(self, *args, **kwargs):
                if self.engine is not None:
                    import gc
                    del self.engine
                    self.engine = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("clean up vllm inference engine from memory")
                else:
                    print("vllm inference engine is not initialized, skip sleep")
            
            def __getattr__(self, item):
                if self.engine is not None:
                    return getattr(self.engine, item)
                else:
                    raise RuntimeError("vllm inference engine is not initialized, cannot get attribute")

        if config.enable_sleep_hack:
            self.inference_engine = _MyInferenceEngineProxy()
        else:
            self.inference_engine = LLM(**vllm_kwargs)

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        # czw: support global-shuffle dispatch for rollout load balancing
        self.shuffle_before_dispatch = config.get("shuffle_before_dispatch", False)
        if self.shuffle_before_dispatch:
            logger.info("vllm rollout will shuffle the prompts before dispatching to workers.")
            self.sampling_params.n = 1

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        
        if "n" in prompts.meta_info:
            kwargs["n"] = prompts.meta_info["n"]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample and (not self.shuffle_before_dispatch):
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch_dict = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if self.shuffle_before_dispatch:
            batch_dict['indices_before_dispatch'] = prompts.batch['indices_before_dispatch']
        batch = TensorDict(batch_dict, batch_size=batch_size,)
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def generate_sequences_textgame(self, env_config, env_num, experiences, tokenizer, num_steps=50, seeds: List[int] = None, validate: bool = False):
        """
        Generate sequences by interacting with textgame environments using vLLM.
        Each environment runs until done or num_steps is reached.
        
        Args:
            env_config: Dict with 'env_id', 'num_players' (without 'num_envs')
            env_num: Number of environments to create
            experiences: List of experience strings, one for each environment
            tokenizer: Tokenizer to use for encoding/decoding
            num_steps: Maximum number of steps per environment (default: 50)
            seeds: Optional list of seeds for each environment. If None, seeds are randomly generated.
                   If provided, must have length equal to env_num.
            
        Returns:
            DataProto containing prompts, responses, and other needed data
        """
        # Ensure inference engine is initialized
        if hasattr(self.inference_engine, "wake_up"):
            self.inference_engine.wake_up()

        env_id = env_config['env_id']
        num_players = env_config['num_players']
        # Optional textgame configs with sensible defaults
        keep_reasoning = env_config['keep_reasoning']
        max_prompt_length = env_config['max_prompt_length']
        max_response_length = env_config['max_response_length']
        no_think = env_config['no_think']
        size = env_config['size'] if 'size' in env_config else None
        num_holes = env_config['num_holes'] if 'num_holes' in env_config else None
        randomize_start_goal = env_config['randomize_start_goal'] if 'randomize_start_goal' in env_config else None
        num_boxes = env_config['num_boxes'] if 'num_boxes' in env_config else None
        dim_room = env_config['dim_room'] if 'dim_room' in env_config else None
        # Create environments inside the worker (avoid Ray serialization issues)
        envs = _make_textgame_envs(env_id=env_id, num_players=num_players, env_num=env_num, size=size, num_holes=num_holes, randomize_start_goal=randomize_start_goal, num_boxes=num_boxes, dim_room=dim_room, seeds=seeds)

        assert len(envs) == len(experiences), f"Number of envs ({len(envs)}) must match number of experiences ({len(experiences)})"

        # Pre-compute sampling kwargs (shared across all envs)
        sampling_kwargs = _build_textgame_sampling_kwargs(max_response_length, validate, self.config.val_kwargs)

        # Store complete trajectories for each environment
        # Format: {env_idx: {'history': [...], 'reward': ..., 'stop_reason': ..., 'final_feedback': ...}}
        env_trajectories = {}
        reward_list = []
        action_dict = _get_textgame_action_dict(env_id)

        num_envs = len(envs)
        # Per-env bookkeeping
        full_observations_list = [{pid: [] for pid in range(num_players)} for _ in range(num_envs)]
        histories = [[] for _ in range(num_envs)]
        done_flags = [False for _ in range(num_envs)]
        step_counters = [0 for _ in range(num_envs)]

        while True:
            # Collect indices of envs that are still active and within step limit
            active_env_indices = [
                env_idx
                for env_idx in range(num_envs)
                if (not done_flags[env_idx]) and (step_counters[env_idx] < num_steps)
            ]

            if len(active_env_indices) == 0:
                break

            # Build batched prompts for all active envs
            vllm_inputs = []
            batch_metadata = []  # store env_idx and current_step_observation_str for each prompt
            for env_idx in active_env_indices:
                env = envs[env_idx]
                experience = experiences[env_idx]
                full_observations = full_observations_list[env_idx]

                # Get current observation from environment
                player_id, observation = env.get_observation()
                if isinstance(observation, str):
                    if len(full_observations[player_id]) > 0:
                        last_observation = full_observations[player_id][-1]
                        # remove the last observation from the observation string
                        observation = observation.split(last_observation[1])[-1]
                    first_game_index = observation.find('[GAME]')
                    last_game_index = observation.rfind('[GAME]')
                    observation = observation[first_game_index:]
                
                # Convert observation to string (LLMObservationWrapper functionality)
                observation_str = _convert_observation_to_string(player_id, observation, full_observations, env)
                current_step_observation_str = _convert_current_step_observation_to_string(player_id, observation, env)
                # Create prompt with experience
                prompt_text = _build_textgame_prompt(experience, observation_str)
                
                prompt_with_template = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=not no_think
                )
                
                # Tokenize the prompt
                prompt_token_ids = _tokenize_textgame_prompt(tokenizer, prompt_with_template, max_prompt_length)

                # Prepare vLLM input for this prompt
                vllm_inputs.append({"prompt_token_ids": prompt_token_ids})
                batch_metadata.append(
                    {
                        "env_idx": env_idx,
                        "step": step_counters[env_idx],
                        "current_step_observation": current_step_observation_str,
                        "prompt_text": prompt_text,
                        "prompt_token_ids": prompt_token_ids,
                        "message": [{"role": "user", "content": prompt_text}],
                    }
                )

            # Prepare LoRA requests if needed
            lora_requests = None
            if self.lora_kwargs:
                lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    lora_int_id = lora_int_ids[0]
                    single_lora_request = LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="/simon-stub-path",
                    )
                    lora_requests = [single_lora_request for _ in range(len(vllm_inputs))]

            # Batched generation using vLLM
            with self.update_sampling_params(**sampling_kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            # Process batched outputs and step environments
            for output, meta in zip(outputs, batch_metadata):
                env_idx = meta["env_idx"]
                if done_flags[env_idx]:
                    # This env might have just been finished in the same batch loop; skip further processing.
                    continue

                env = envs[env_idx]

                # Get generated response tokens (n=1)
                response_ids = output.outputs[0].token_ids

                # Decode the generated action
                raw_response = tokenizer.decode(response_ids, skip_special_tokens=True)

                # Optionally strip out reasoning before </think>, controlled by trainer.textgame_keep_reasoning
                raw_response = _postprocess_textgame_response(raw_response, keep_reasoning)

                # Extract action from response (handles cases with reasoning text)
                if 'Minesweeper' in env_id:
                    # For Minesweeper, extract the last "[row col]" style coordinate pair
                    extracted_action = _extract_minesweeper_action_from_response(raw_response)
                else:
                    extracted_action = _extract_action_from_response(raw_response, action_dict)

                # Format action (add brackets if missing)
                action_text = _format_action(extracted_action)

                # Execute action in environment
                done, step_info = env.step(action=action_text)
                done_flags[env_idx] = done

                # Store this step's data in history (as text, not token ids)
                histories[env_idx].append(
                    {
                        'step': meta["step"],
                        'prompt_text': meta["prompt_text"],
                        'prompt_token_ids': meta["prompt_token_ids"],
                        "message": meta["message"],
                        'raw_response': raw_response,
                        'response_ids': response_ids,
                        'current_step_observation': meta["current_step_observation"],
                    }
                )

                step_counters[env_idx] += 1

        # Close environments and collect results
        for env_idx, env in enumerate(envs):
            rewards, game_info = env.close()
            done = done_flags[env_idx]
            history = histories[env_idx]

            # if reward is None, set it to a dict of 0 # only when the env is not done, reward is None
            if rewards is None:
                rewards = {0: 0}
            # set all rewards < 1.0 to be 0
            for player_id in rewards.keys():
                if rewards[player_id] < 1.0:
                    rewards[player_id] = 0
            if done:
                if 'Sokoban' in env_id:
                    result = "Success" if rewards[0] == 1.0 else "Failure"
                    final_feedback = (
                    "Summary:\n"
                    "Final Result: {}\n"
                    "Feedback: {}\n").format(result, game_info[0]['reason'])
                elif 'FrozenLake' in env_id:
                    result = "Success" if rewards[0] == 1.0 else "Failure"
                    final_feedback = (
                    "Summary:\n"
                    "Final Result: {}\n"
                    "Feedback: {}\n").format(result, game_info[0]['reason'])
                else:
                    assert False, f"env: {env_id} summary not implemented"
            else:
                final_feedback = "Environment terminated because the round limit is reached"

            # Determine stop reason
            stop_reason = "done" if done else "max_steps"
            turn_count = game_info[0]['turn_count']

            # Store trajectory for this environment
            env_trajectories[env_idx] = {
                'history': history,
                'reward': rewards,
                'turn_count': turn_count,
                'stop_reason': stop_reason,
                'final_feedback': final_feedback
            }
            # Add reward to list
            reward_list.append(rewards)
        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        return {
            'env_trajectories': env_trajectories,
            'reward_list': reward_list
        }


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
