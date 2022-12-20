from datasets import load_dataset

from . import BaseTask

class HellaSwag(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('hellaswag')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 4
        self.preprocess_dataset()
    
    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["ctx"]
        answer_str = []
        for i in range(self.class_num):
            answer_str.append(' ' + example["endings"][i])
        label = int(example["label"])
        return input_str, answer_str, label
