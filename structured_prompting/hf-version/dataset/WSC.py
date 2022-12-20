from datasets import load_dataset

from . import BaseTask

class WSC(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'wsc')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.templates = self.templates_set_without_newline()
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("{text} In the previous sentence, does the pronoun \"{span2_text}\" refer to {span1_text}?", " {answer}", ["No", "Yes"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", example["text"]).replace("{span2_text}", example["span2_text"]).replace("{span1_text}", example["span1_text"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

