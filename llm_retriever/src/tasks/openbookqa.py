from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("openbookqa")
class Openbookqa(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('openbookqa', 'additional', split=split)

        dataset = dataset.rename_column('fact1', 'fact')
        dataset = dataset.rename_column('question_stem', 'question')

        def _map_func(ex: Dict) -> Dict:
            ex['options'] = ex['choices']['text']
            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{fact} {question}", "{answer}"),
            ("Read this fact: \"{fact}\" Now answer this question: \"{question}\"", "{answer}"),
            ("Given the fact \"{fact}\", what is the answer to the question or completion \"{question}\"", "{answer}"),
            ("Knowing that \"{fact}\", how would one answer \"{question}\"", "{answer}"),
            ("Use evidence from the fact that {fact} to answer this question: \"{question}\"", "{answer}"),
            ("Fact: {fact} Question: {question} What's the answer?", "{answer}"),
            ("Use this fact to answer the question: {fact} {question}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B', 'C', 'D']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'openbookqa'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return to_letter(example['answerKey'])
