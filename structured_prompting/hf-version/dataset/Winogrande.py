from datasets import load_dataset

from . import BaseTask

class Winogrande(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('winogrande', 'winogrande_xs')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        cut_index = example["sentence"].index('_')
        text_first = example["sentence"][:cut_index]
        text_second = example["sentence"][cut_index+1:]
        input_str = text_first
        answer_str = [" " + example["option1"] + text_second, " " + example["option2"] + text_second]
        label = int(example["answer"]) - 1
        return input_str, answer_str, label