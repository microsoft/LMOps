from datasets import load_dataset

from . import BaseTask

class AGNews(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('ag_news')
        self.dataset = dataset['train'] if is_train else dataset['test']
        self.class_num = 4
        self.preprocess_dataset()
        

    def templates_set_without_newline(self):
        return [
            ("Article: {text} Category:", " {answer}", ["World", "Sports", "Business", "Technology"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", example["text"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label
