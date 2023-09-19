from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("copa")
class Copa(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('super_glue', 'copa', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['options'] = [ex['choice1'], ex['choice2']]
            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        # question is either "cause" or "effect"
        return [
            ("\"{premise}\" What is the {question}?", "{answer}"),
            ("Here is a premise: \"{premise}\" What is the {question}?", "{answer}"),
            ("\"{premise}\" What is the {question} of the preceding sentence?", "{answer}"),
            ("\"{premise}\" What is a plausible {question}?", "{answer}"),
            ("Based on the following sentence, what is the {question}? \"{premise}\"", "{answer}"),
            ("\"{premise}\" {question}:", "{answer}"),
            ("What is the {question} of the following sentence? \"{premise}\"", "{answer}"),
            ("Answer the following question about this sentence: \"{premise}\" What is the {question}?", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'copa'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return to_letter(str(example['label']))
