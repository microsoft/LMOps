from datasets import load_dataset

from . import BaseTask

class WiC(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'wic')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.templates = self.templates_set_without_newline()
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("{sentence1} {sentence2} Question: Is the word \"{word}\" used in the same way in the two sentences above? Answer:", " {answer}", ["No", "Yes"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{word}", example["word"]).replace("{sentence1}", example["sentence1"]).replace("{sentence2}", example["sentence2"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label