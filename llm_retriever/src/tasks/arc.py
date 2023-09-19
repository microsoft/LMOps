from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("arc_c")
class Arc_c(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('ai2_arc', 'ARC-Challenge', split=split)

        # Both FLAN & uprise & structured prompting have this filter logic
        dataset = dataset.filter(lambda ex: len(ex['choices']['text']) == 4)

        def _map_func(ex: Dict) -> Dict:
            if ex['answerKey'] not in ['A', 'B', 'C', 'D']:
                ex["answerKey"] = to_letter(int(ex['answerKey']) - 1)
            ex['options'] = ex['choices']['text']
            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{question}", "{answer}"),
            ("Question: {question} Answer:", "{answer}"),
            ("Question: {question} What is the correct answer to the question from the following choices?", "{answer}"),
            ("Q: {question} What is the correct answer to this question?", "{answer}"),
            ("What is the answer? {question}", "{answer}"),
            ("Answer the question {question}", "{answer}"),
            ("{question} Pick the answer from these options.", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B', 'C', 'D']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'arc_c'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['answerKey']


@task_map.add("arc_e")
class Arc_e(Arc_c):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('ai2_arc', 'ARC-Easy', split=split)
        dataset = dataset.filter(lambda ex: len(ex['choices']['text']) == 4)

        def _map_func(ex: Dict) -> Dict:
            if ex['answerKey'] not in ['A', 'B', 'C', 'D']:
                ex["answerKey"] = to_letter(int(ex['answerKey']) - 1)
            ex['options'] = ex['choices']['text']
            return ex

        dataset = dataset.map(_map_func)
        return dataset

    @property
    def task_name(self) -> str:
        return 'arc_e'
