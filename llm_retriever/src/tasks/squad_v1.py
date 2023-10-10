import re

from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("squad_v1")
class Squad_v1(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('squad', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['title'] = re.sub(r'_', ' ', ex['title'])
            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Please answer a question about the following article about {title}: {context} {question}", "{answer}"),
            ("Read this and answer the question {context} {question}", "{answer}"),
            ("{context} {question}", "{answer}"),
            ("Answer a question about this article: {context} {question}", "{answer}"),
            ("Here is a question about this article: {context} What is the answer to this question: {question}",
             "{answer}"),
            ("Article: {context} Question: {question}", "{answer}"),
            ("Article: {context} Now answer this question: {question}", "{answer}"),
            ("{title} {context} Q: {question}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'squad'

    @property
    def task_name(self) -> str:
        return 'squad_v1'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['answers']['text']
