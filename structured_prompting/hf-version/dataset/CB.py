from datasets import load_dataset

from . import BaseTask

class CB(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'cb')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 3
        self.preprocess_dataset()
        

    def templates_set_without_newline(self):
        return [
            ("Read the following paragraph and determine if the hypothesis is true: {premise} Hypothesis: {hypothesis}. ", "{answer}", ["Yes", "No", "Maybe"]),
            ("{premise} Based on the paragraph above can we conclude that \"{hypothesis}\"? Yes, No, or Maybe? Answer: ", " {answer}", ["Yes", "No", "Maybe"]),
            ("{premise} Can we infer the following? {hypothesis}. ", "{answer}", ["Yes", "No", "Maybe"]),
            ("{premise} Question: {hypothesis}. True, False, or Neither? Answer: ", "{answer}", ["True", "False", "Neither"]),
            ("Can we draw the following hypothesis from the context? Context: {premise} Hypothesis: {hypothesis}. Answer: ", "{answer}", ["Yes", "No", "Maybe"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{premise}", example["premise"]).replace("{hypothesis}", example["hypothesis"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label


