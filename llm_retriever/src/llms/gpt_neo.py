from logger_config import logger
from config import Arguments
from llms.gpt2 import GPT2


class GPTNeo(GPT2):

    def __init__(self, args: Arguments, model_name_or_path: str = 'EleutherAI/gpt-neo-2.7B', **kwargs):
        super().__init__(args, model_name_or_path, **kwargs)
