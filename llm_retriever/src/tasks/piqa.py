from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("piqa")
class Piqa(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('piqa', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['options'] = [ex['sol1'], ex['sol2']]
            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Here is a goal: \"{goal}\" How would you accomplish this goal?", "{answer}"),
            ("Here is a goal: \"{goal}\" Which way makes more sense to accomplish this goal?", "{answer}"),
            ("Goal: \"{goal}\" Which of the following methods is more reasonable for accomplishing this goal?",
             "{answer}"),
            ("BaseTaskive: \"{goal}\" Which of the following solutions is more sound in terms of naive physics reasoning?",
             "{answer}"),
            ("How do you do this: \"{goal}\"", "{answer}"),
            ("What is the best way to: \"{goal}\"", "{answer}"),
            ("Which of the following solutions is better for the following goal: \"{goal}\"", "{answer}"),
            ("How would someone go about accomplishing this goal? \"{goal}\"", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'piqa'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return to_letter(example['label'])
