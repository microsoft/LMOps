from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("multirc")
class Multirc(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('super_glue', 'multirc', split=split)
        dataset = dataset.rename_column('answer', 'response')

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            (
                "{paragraph} Question: \"{question}\" Response: \"{response}\" Does the response correctly answer the question?",
                "{answer}"),
            (
                "{paragraph} Question: \"{question}\" Response: \"{response}\" Based on the paragraph, is the response to the question is factually correct?",
                "{answer}"),
            ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer correct?", "{answer}"),
            (
                "Paragraph: {paragraph} Question: \"{question}\" Answer: \"{response}\" Based on the paragraph, is this answer correct",
                "{answer}"),
            (
                "{paragraph} Based on the paragraph, does the response \"{response}\" correctly answer the question \"{question}\"?",
                "{answer}"),
            (
                "{paragraph} According to the above paragraph, the correct answer to the question \"{question}\" is \"{response}\"?",
                "{answer}"),
            (
                "{paragraph} After reading the above, is \"{response}\" the correct answer to the question \"{question}\"?",
                "{answer}"),
            ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer to the question correct?",
             "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'f1'

    @property
    def task_name(self) -> str:
        return 'multirc'
