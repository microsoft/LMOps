from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


# define your task class
@task_map.add("snli")  # add your task to the task map
class Snli(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('snli', split=split)
        dataset = dataset.filter(lambda ex: int(ex["label"]) >= 0)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("If \"{premise}\", does this mean that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ("If \"{premise}\", can we conclude \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ("If \"{premise}\", does it logically follow that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            (
                "Based on the sentence \"{premise}\", is the sentence \"{hypothesis}\" a true sentence? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Premise: {premise} Hypothesis: {hypothesis} Can we conclude that the hypothesis is true if the premise is true? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Premise: {premise} Hypothesis: {hypothesis} Given the premise, can we conclude the hypothesis? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Here is a premise: \"{premise}\" Here is a hypothesis: \"{hypothesis}\". Does the premise tell us whether the hypothesis is true? Yes, No, or Maybe?",
                "{answer}"),
            ("Is it possible to conclude that \"{premise}\" if \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ("Is the premise \"{premise}\" true if \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
        ]


    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Yes', 'Maybe', 'No']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'snli'
