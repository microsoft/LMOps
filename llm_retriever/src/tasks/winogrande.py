from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("winogrande")
class Winogrande(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('winogrande', 'winogrande_xl', split=split)

        def _map_func(ex: Dict) -> Dict:
            cut_index = ex["sentence"].index('_')
            context = ex["sentence"][:cut_index]
            ex['context'] = context.strip()

            text_second = ex["sentence"][cut_index + 1:]
            ex['options'] = [ex["option1"] + text_second, ex["option2"] + text_second]

            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("How does the sentence end? {context}", "{answer}"),
            ("Write the next sentence. {context}", "{answer}"),
            ("Continue the following story. {context}", "{answer}"),
            ("Complete the following sentence. {context}", "{answer}"),
            ("Continue writing the following text. {context}", "{answer}"),
            ("How does the sentence end? {context}", "{answer}"),
            ("Write the next sentence. {context}", "{answer}"),
            ("Continue the following story. {context}", "{answer}"),
            ("Complete the following sentence. {context}", "{answer}"),
            ("Continue writing the following text. {context}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'winogrande'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        label = int(example['answer']) - 1
        return to_letter(label)
