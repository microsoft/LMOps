from typing import Optional, List, Tuple, Dict
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("qqp")
class Qqp(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('glue', 'qqp', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['question1'] = ex['question1'].replace('""', '\'')
            ex['question2'] = ex['question2'].replace('""', '\'')
            return ex

        dataset = dataset.map(_map_func)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("\"{question1}\" \"{question2}\" Would you say that these questions are the same?", "{answer}"),
            ("\"{question1}\" \"{question2}\" Do those questions have the same meaning?", "{answer}"),
            ("\"{question1}\" \"{question2}\" Are these two questions inquiring about the same information?", "{answer}"),
            ("\"{question1}\" \"{question2}\" Please tell me if those questions are the same.", "{answer}"),
            ("\"{question1}\" \"{question2}\" Are these two questions paraphrases of each other?", "{answer}"),
            ("First question: \"{question1}\" Second question: \"{question2}\" Are these two questions asking the same thing?",
             "{answer}"),
            (
                "Question 1: \"{question1}\" Question 2: \"{question2}\" Are questions 1 and 2 asking the same thing?",
                "{answer}"),
            ("Question 1: \"{question1}\" Question 2: \"{question2}\" Would the answer to these two questions be the same?",
             "{answer}"),
            ("Are the following two questions the same? \"{question1}\" \"{question2}\"", "{answer}"),
            ("Do these questions have the same meaning? \"{question1}\" \"{question2}\"", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'acc_and_f1'

    @property
    def task_name(self) -> str:
        return 'qqp'
