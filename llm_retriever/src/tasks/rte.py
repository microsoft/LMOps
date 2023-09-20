from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("rte")
class Rte(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('super_glue', 'rte', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{premise} Based on the paragraph above can we conclude that \"{hypothesis}\"? Yes or No?",
             "{answer}"),
            (
                "{premise} Based on that paragraph can we conclude that this sentence is true? {hypothesis} Yes or No?",
                "{answer}"),
            ("{premise} Can we draw the following conclusion? {hypothesis} Yes or No?", "{answer}"),
            ("{premise} Does this next sentence follow, given the preceding text? {hypothesis} Yes or No?",
             "{answer}"),
            ("{premise} Can we infer the following? {hypothesis} Yes or No?", "{answer}"),
            (
                "Read the following paragraph and determine if the hypothesis is true: {premise} Hypothesis: {hypothesis} Yes or No?",
                "{answer}"),
            ("Read the text and determine if the sentence is true: {premise} Sentence: {hypothesis} Yes or No?",
             "{answer}"),
            (
                "Can we draw the following hypothesis from the context?  Context: {premise} Hypothesis: {hypothesis} Yes or No?",
                "{answer}"),
            ("Determine if the sentence is true based on the text below: {hypothesis} {premise} Yes or No?",
             "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Yes', 'No']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'rte'
