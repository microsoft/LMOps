"""OpenAI-compatible API client used as a reward model in ray_trainer."""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import openai
from openai import OpenAI


MODEL_CONFIGS = {
    "gpt4o": {
        "default_model": "gpt-4o",
        "default_max_workers": 128,
        "use_max_completion_tokens": False,
    },
    "gpt54": {
        "default_model": "gpt-5.4",
        "default_max_workers": 128,
        "use_max_completion_tokens": True,
    },
    "gpt55": {
        "default_model": "gpt-5.5",
        "default_max_workers": 128,
        "use_max_completion_tokens": True,
    },
}


def _first_env(*names: str):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class GptRMClient:
    """Threaded reward-model client configured entirely through environment variables."""

    def __init__(self, exp_model_path: str, max_workers: int = None):
        """
        Args:
            exp_model_path: ``gpt4o`` or a reasoning model with an optional
                effort suffix, for example ``gpt54_high``.
        """
        parts = exp_model_path.split("_")
        self.model_key = parts[0]
        self.reasoning_effort = parts[1] if len(parts) > 1 else None

        if self.model_key not in MODEL_CONFIGS:
            supported = ", ".join(MODEL_CONFIGS)
            raise ValueError(f"Unknown GPT reward model '{self.model_key}'. Supported: {supported}")

        config = MODEL_CONFIGS[self.model_key]
        self.use_max_completion_tokens = config["use_max_completion_tokens"]
        env_workers = os.environ.get("GPT_RM_MAX_WORKERS")
        self.max_workers = max_workers or (int(env_workers) if env_workers else config["default_max_workers"])
        timeout = float(os.environ.get("GPT_RM_TIMEOUT", "400"))

        model_env_name = f"GPT_RM_MODEL_{self.model_key.upper()}"
        self.model = _first_env(model_env_name, "GPT_RM_MODEL", "OPENAI_MODEL") or config["default_model"]

        api_key = os.environ.get("OPENAI_API_KEY")
        fallback_api_key = os.environ.get("GPT_RM_API_KEY")
        if not api_key and not fallback_api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using the GPT reward model.")

        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("GPT_RM_BASE_URL")
        client_kwargs = {"timeout": timeout, "max_retries": 0}
        if not api_key:
            client_kwargs["api_key"] = fallback_api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        endpoint_label = base_url or "https://api.openai.com/v1"

        self.clients = [(client, self.model, endpoint_label)]
        self.base_weights = [1.0]
        self.cooldown_until = [0.0]
        print(
            "[GptRMClient] Ready: "
            f"model={self.model}, max_workers={self.max_workers}"
        )

    def _get_weights(self):
        """Return endpoint weights, excluding endpoints in cooldown."""
        now = time.time()
        weights = [
            0.0 if now < self.cooldown_until[i] else float(weight)
            for i, weight in enumerate(self.base_weights)
        ]
        return weights if sum(weights) else [float(weight) for weight in self.base_weights]

    def call(self, prompt: str, max_tokens: int = 8192) -> str:
        """Call the configured reward model with one prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if self.use_max_completion_tokens:
            kwargs = {
                "messages": messages,
                "temperature": 1.0,
                "max_completion_tokens": max_tokens,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }
        else:
            kwargs = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }

        if self.reasoning_effort:
            if self.model_key == "gpt54":
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
            else:
                kwargs["reasoning_effort"] = self.reasoning_effort

        max_retry = 3
        cur_retry = 0
        consecutive_fails = {}
        while cur_retry <= max_retry:
            weights = self._get_weights()
            idx = random.choices(range(len(self.clients)), weights=weights, k=1)[0]
            client, model, endpoint = self.clients[idx]

            try:
                completion = client.chat.completions.create(model=model, **kwargs)
                consecutive_fails.pop(idx, None)
                usage = completion.usage
                if usage and usage.completion_tokens >= max_tokens:
                    print(
                        f"[GptRM] WARNING: completion_tokens ({usage.completion_tokens}) "
                        f">= max_tokens ({max_tokens}), response may be truncated"
                    )
                return completion.choices[0].message.content or ""
            except openai.RateLimitError:
                sleep_time = 2**cur_retry
                print(f"[GptRM] RateLimitError on {endpoint[:40]}, sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                cur_retry += 1
            except Exception as exc:
                error = str(exc)
                if "ResponsibleAIPolicyViolation" in error:
                    print("[GptRM] ResponsibleAIPolicyViolation, skipping")
                    return ""
                consecutive_fails[idx] = consecutive_fails.get(idx, 0) + 1
                sleep_time = 2**cur_retry
                print(
                    f"[GptRM] Retry {cur_retry} on {endpoint[:40]}: "
                    f"{error[:200]}, sleeping {sleep_time}s..."
                )
                time.sleep(sleep_time)
                cur_retry += 1
                if consecutive_fails[idx] >= 5:
                    self.cooldown_until[idx] = time.time() + 500
                    print(f"[GptRM] Endpoint {endpoint[:40]} cooled down for 500s")

        print(f"[GptRM] Max retries ({max_retry}) reached, skipping")
        return ""

    def batch_call(self, prompts: List[str], max_tokens: int = 8192) -> List[str]:
        """Call the reward model concurrently and preserve input order."""
        results = [""] * len(prompts)

        def _call_single(idx_prompt):
            idx, prompt = idx_prompt
            return idx, self.call(prompt, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_call_single, (idx, prompt)): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response

        return results
