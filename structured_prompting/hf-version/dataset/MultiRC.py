from datasets import load_dataset

from . import BaseTask

class MultiRC(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'multirc')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.templates = self.templates_set_without_newline()
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("{paragraph} Question: \"{question}\" Response: \"{response}\" Does the response correctly answer the question? Answer:", " {answer}", ["No", "Yes"]),
            ("{paragraph} Question: \"{question}\" Response: \"{response}\" Based on the paragraph, is the response to the question is factually correct?", " {answer}", ["No", "Yes"]),
            ("{paragraph} Based on the paragraph, does the response \"{response}\" correctly answer the question \"{question}\"?", " {answer}", ["No", "Yes"]),
            ("{paragraph} According to the above paragraph, the correct answer to the question \"{question}\" is \"{response}\"? Answer:", " {answer}", ["No", "Yes"]),
            ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer to the question True or False?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{paragraph}", example["paragraph"]).replace("{question}", example["question"]).replace("{response}", example["answer"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label
        