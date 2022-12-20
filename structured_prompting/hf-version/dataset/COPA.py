from datasets import load_dataset

from . import BaseTask

class COPA(BaseTask):
    def __init__(self, is_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'copa')
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 2
        self.preprocess_dataset()
        

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        def copa_answer(string):
            string = ' ' + string[0].lower() + string[1:]
            return string

        text_first = example["premise"]

        if self.temp_index == 0:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " because"
            else:
                text_first = text_first + " so"
            input_str = text_first
            answer_str = [copa_answer(example["choice1"]), copa_answer(example["choice2"])]
        elif self.temp_index == 2:
            input_str = [text_first + " What is the " + example["question"] + "?"] * self.class_num
            answer_str = [' ' + example["choice1"], ' ' + example["choice2"]]
        elif self.temp_index == 3:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " because"
            else:
                text_first = text_first + " so"
            input_str = text_first
            answer_str = [' '+example["choice1"], ' '+example["choice2"]]
        elif self.temp_index == 4:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " This happened because"
            else:
                text_first = text_first + " As a consequence,"
            input_str = text_first
            answer_str = [copa_answer(example["choice1"]), copa_answer(example["choice2"])]
        label = example["label"]
        return input_str, answer_str, label

