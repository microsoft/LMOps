from datasets import load_dataset

from . import BaseTask

class BoolQ(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'boolq')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("Passage: {passage} After reading this passage, I have a question: {question}? True or False?", " {answer}", ["False", "True"]),
            ("Text: {passage} Question: {question}? Answer:", " {answer}", ["No", "Yes"]),
            ("{passage} Based on the above text, what's the best answer to this question: {question}?", " {answer}", ["No", "Yes"]),
            ("Based on the following passage, {question}? {passage} Please answer yes or no.", " {answer}", ["No", "Yes"]),
            ("Exercise: read the text and answer the question by True or False. Text: {passage} Question: {question}?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{passage}", example["passage"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

