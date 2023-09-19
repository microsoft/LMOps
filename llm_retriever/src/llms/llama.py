from logger_config import logger
from config import Arguments
from llms.gpt2 import GPT2


class Llama(GPT2):

    def __init__(self, args: Arguments, model_name_or_path: str = 'huggyllama/llama-7b', **kwargs):
        super().__init__(args, model_name_or_path, **kwargs)
