from datasets import load_dataset

from . import BaseTask

class TREC(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('trec')
        self.dataset = dataset['train'] if is_train else dataset['test']
        self.class_num = 6
        self.preprocess_dataset()
        

    def templates_set_without_newline(self):
        return [
            ("Question: {text} Type:", " {answer}", ["Description", "Entity", "Expression", "Person", "Number", "Location"])
        ]


    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", example["text"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["coarse_label"]
        return input_str, answer_str, label

