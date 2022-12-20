from datasets import load_dataset

from . import BaseTask

class DBPedia(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("dbpedia_14")
        self.dataset = dataset['train'] if is_train else dataset['test']
        self.class_num = 14
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("Input: {content} Type:", " {answer}", ["company", "school", "artist", "sport", "politics", "transportation", "building", "nature", "village", "animal", "plant", "album", "film", "book"])
        ]

    def preprocess_example(self, example):
        content = example["content"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{content}", content)
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label
        