from typing import List, Union, Optional


def to_letter(key: Union[str, int]) -> str:
    key = str(key).upper().strip()
    num_to_letter = {"0": "A", "1": "B", "2": "C", "3": "D"}
    assert key in num_to_letter or key in ['A', 'B', 'C', 'D'], f'Unknown answer key: {key}'
    return num_to_letter.get(key, key)


def format_options(options: List[str]) -> str:
    assert len(options) <= 4, f'Number of options should be less than 4, but got {len(options)}'
    res = 'OPTIONS: '
    for letter, option in zip(['A', 'B', 'C', 'D'], options):
        res += f'{letter}) {option} '

    return res.strip()


# Based on https://github.com/microsoft/LMOps/blob/main/uprise/DPR/dpr/utils/tasks.py

TASK_TYPE_TO_TASK_NAME = {
    'close_qa': ['natural_questions', 'arc_c', 'arc_e'],
    'common_reason': ['copa', 'piqa', 'hellaswag'],
    'coreference': ['winogrande', 'wsc', 'wsc273'],
    'nli': ['rte', 'mnli', 'mnli_m', 'mnli_mm', 'qnli', 'snli'],
    'paraphrase': ['mrpc', 'paws', 'qqp'],
    'reading': ['multirc', 'openbookqa', 'squad_v1', 'boolq'],
    'sentiment': ['yelp', 'sentiment140', 'sst2'],
    'struct2text': ['common_gen', 'e2e_nlg', 'dart'],
    'summarize': ['aeslc', 'ag_news', 'gigaword'],
}


class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls

        return adder


task_map = App()


from .base_task import BaseTask
from .aeslc import Aeslc
from .agnews import Ag_news
from .arc import Arc_c, Arc_e
from .boolq import Boolq
from .common_gen import Common_gen
from .copa import Copa
from .dart import Dart
from .e2e_nlg import E2e_nlg
from .gigaword import Gigaword
from .hellaswag import Hellaswag
from .mnli import Mnli, Mnli_m, Mnli_mm
from .mrpc import Mrpc
from .multirc import Multirc
from .nq import Natural_questions
from .openbookqa import Openbookqa
from .paws import Paws
from .piqa import Piqa
from .qnli import Qnli
from .qqp import Qqp
from .rte import Rte
from .sentiment140 import Sentiment140
from .snli import Snli
from .squad_v1 import Squad_v1
from .sst2 import Sst2
from .winogrande import Winogrande
from .wsc import Wsc
from .wsc273 import Wsc273
from .yelp import Yelp


def get_metric_name_by_task_name(task_name: str) -> str:
    assert task_name in task_map.cls_dic, f'Unknown task name: {task_name}'
    return task_map.cls_dic[task_name]().metric_name


def get_possible_answers_by_task_name(task_name: str) -> Optional[List[str]]:
    assert task_name in task_map.cls_dic, f'Unknown task name: {task_name}'
    return task_map.cls_dic[task_name]().possible_answers


def parse_decoded_text_by_task(decoded_text: str, task_name: str) -> str:
    # TODO: maybe add some task-specific logics here
    return decoded_text.strip().split('\n')[0]
