from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("mnli")
class Mnli(BaseTask):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        if split != 'train':
            return None
        dataset: Dataset = load_dataset('glue', 'mnli', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            (
                "Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Does the premise entail the hypothesis? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Is the hypothesis entailed by the premise? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Here is a premise: \"{premise}\" Here is a hypothesis: \"{hypothesis}\" Is it possible to conclude that if the premise is true, then so is the hypothesis? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Sentence 1: \"{premise}\" Sentence 2: \"{hypothesis}\" Is this second sentence entailed by the first sentence? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Sentence 1: \"{premise}\" Sentence 2: \"{hypothesis}\" If the first sentence is true, then is the second sentence true? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Based on the premise \"{premise}\", can we conclude the hypothesis \"{hypothesis}\" is true? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Premise: \"{premise}\" If this premise is true, what does that tell us about whether it entails the hypothesis \"{hypothesis}\"? Yes, No, or Maybe?",
                "{answer}"),
            (
                "Premise: \"{premise}\" Based on this premise, is the hypothesis \"{hypothesis}\" true? Yes, No, or Maybe?",
                "{answer}"),
            ("If \"{premise}\", can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ("\"{premise}\" Does it follow that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Yes', 'Maybe', 'No']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'mnli'


@task_map.add("mnli_m")
class Mnli_m(Mnli):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        if split == 'train':
            return None

        return load_dataset('glue', 'mnli_matched', split='validation')

    @property
    def task_name(self) -> str:
        return 'mnli_m'


@task_map.add("mnli_mm")
class Mnli_mm(Mnli):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        if split == 'train':
            return None

        return load_dataset('glue', 'mnli_mismatched', split='validation')

    @property
    def task_name(self) -> str:
        return 'mnli_mm'
