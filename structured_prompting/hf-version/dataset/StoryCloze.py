from datasets import load_dataset

from . import BaseTask

class StoryCloze(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('story_cloze', '2016', data_dir="./data/StoryCloze")
        self.dataset = dataset['validation'] if is_train else dataset['test']
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["input_sentence_1"] + ' ' + example["input_sentence_2"] + ' ' + example["input_sentence_3"] + ' ' + example["input_sentence_4"]
        answer_str = [' '+example["sentence_quiz1"], ' '+example["sentence_quiz2"]]
        label = int(example["answer_right_ending"]) - 1
        return input_str, answer_str, label