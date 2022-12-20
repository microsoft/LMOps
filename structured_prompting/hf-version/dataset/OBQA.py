from datasets import load_dataset

from . import BaseTask

class OBQA(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('openbookqa', 'main')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 4
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["question_stem"]
        answer_str = []
        for i in range(4):
            answer_str.append(' '+example["choices"]["text"][i])
        
        if example["answerKey"] == 'A':
            label = 0
        elif example["answerKey"] == 'B':
            label = 1
        elif example["answerKey"] == 'C':
            label = 2
        elif example["answerKey"] == 'D':
            label = 3
        return input_str, answer_str, label