from datasets import load_dataset

from . import BaseTask

class PIQA(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('piqa')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["goal"]
        answer_str = [' '+example["sol1"], ' '+example["sol2"]]
        label = example["label"]
        return input_str, answer_str, label