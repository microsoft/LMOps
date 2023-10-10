from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("qnli")
class Qnli(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('glue', 'qnli', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Does the sentence \"{sentence}\" answer the question \"{question}\"?", "{answer}"),
            ("Does the sentence \"{sentence}\" provide a valid answer to the question \"{question}\"?", "{answer}"),
            ("Is \"{sentence}\" a good answer to the question \"{question}\"?", "{answer}"),
            ("Does \"{sentence}\" correctly answer the question of \"{question}\"?", "{answer}"),
            ("Does \"{sentence}\" contain the correct answer to \"{question}\"?", "{answer}"),
            ("Q: {question}  A: {sentence}  Does the answer correctly answer the question?", "{answer}"),
            (
                "Question: {question} Answer: {sentence}  Is the question answered in a satisfactory fashion?",
                "{answer}"),
            ("Question: {question} Is {sentence} a good answer to this question?", "{answer}"),
            ("Question: {question} Is \"{sentence}\" the correct answer?", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Yes', 'No']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'qnli'
