from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("natural_questions")
class Natural_questions(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('nq_open', split=split)
        dataset = dataset.map(lambda ex: {'question': ex['question'] + '?'})
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Question: {question} Answer:", "{answer}"),
            ("{question}", "{answer}"),
            ("Answer the following question: {question}", "{answer}"),
            ("Answer this question: {question}", "{answer}"),
            ("Please answer this question: {question}", "{answer}"),
            ("Answer the question...{question}", "{answer}"),
            ("What is the answer to this question? {question}", "{answer}"),
            ("Can you tell me the answer to {question}", "{answer}"),
            ("Next question: {question}", "{answer}"),
            ("Q: {question} A:", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'trivia_qa'

    @property
    def task_name(self) -> str:
        return 'natural_questions'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['answer']
