from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai.types.chat import ChatCompletion

from utils import AtomicCounter


def get_vllm_model_id(host: str = "localhost", port: int = 8000, api_key: str = "token-123") -> str:
    openai_api_base = f"http://{host}:{port}/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    return models.data[0].id


class VllmClient:

    def __init__(self, model: str, host: str = 'localhost', port: int = 8000, api_key: str = 'token-123'):
        super().__init__()
        self.model = model
        self.client: OpenAI = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
        )
        self.token_consumed: AtomicCounter = AtomicCounter()

    def call_chat(self, messages: List[Dict], return_str: bool = True, **kwargs) -> Union[str, ChatCompletion]:
        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        self.token_consumed.increment(num=completion.usage.total_tokens)

        return completion.choices[0].message.content if return_str else completion

    def batch_call_chat(self, messages: List[List[Dict]], return_str: bool = True, num_workers: int = 4, **kwargs) -> List[Union[str, ChatCompletion]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(
                lambda m: self.call_chat(m, return_str=return_str, **kwargs),
                messages
            ))
