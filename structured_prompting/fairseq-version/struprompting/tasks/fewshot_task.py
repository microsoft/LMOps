import datasets
import random
import torch
import json
import string 
import numpy as np
import copy
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score

class BaseTask(object):
    def __init__(self, tokenizer, dictionary, seed=1, k=2, temp_index=0, gpt_maxlen=2048, permut_index=-1, chunk_k=16, chunk_len=-1, truncate=False, pad_space=False):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.tokenizer = tokenizer
        self.dictionary = dictionary

        self.k = k 
        self.temp_index = temp_index
        self.class_num = 1
        self.gpt_maxlen = gpt_maxlen

        self.permut_index = permut_index
        self.chunk_k = chunk_k
        self.chunk_len = chunk_len
        self.truncate = truncate
        self.pad_space = pad_space

        self.answer_set = []

    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def get_dataset_train(self, data):
        return data.shuffle(seed=self.seed)
    
    def get_dataset_valid(self, data):
        res = data.shuffle(seed=1)
        if len(res) > 4000:
            res = res.select([i for i in range(4000)])
        return res

    def get_data_for_fewshot(self):
        src_tokens_train, _, _, _ = self.tokenized_data('train')
        src_tokens_valid, gpt_loss_mask_valid, labels, max_valid_len = self.tokenized_data('valid')

        permutation_set = []
        min_num = min(len(src_tokens_train), self.k)
        if self.permut_index > 0:
            for i in range(5):
                permutation_set.append(np.random.permutation(min_num))
            src_tokens_train = src_tokens_train[permutation_set[self.permut_index]]

        context_src_tokens = [0]
        true_k = 0
        for i in range(len(src_tokens_train)):
            if len(context_src_tokens) + len(src_tokens_train[i][1:]) + max_valid_len < self.gpt_maxlen:
                context_src_tokens.extend(src_tokens_train[i][1:])
                true_k = i + 1
            else:
                true_k = i
                break
        
        print(f"NOTE: true K of baseline is {true_k}, max valid len is {max_valid_len}")

        for i in range(len(labels) // self.class_num):
            for j in range(i*self.class_num, (i+1)*self.class_num):
                src_tokens_valid[j] = context_src_tokens + src_tokens_valid[j][1:]
                gpt_loss_mask_valid[j] = [False]*len(context_src_tokens) + gpt_loss_mask_valid[j][1:]

        return src_tokens_valid, gpt_loss_mask_valid, labels, self.answer_set


    def get_data_for_manyshot(self):
        src_tokens_train, _, _, _ = self.tokenized_data('train')
        src_tokens_valid, gpt_loss_mask_valid, labels, max_valid_len = self.tokenized_data('valid')

        permutation_set = []
        min_num = min(len(src_tokens_train), self.k)
        if self.permut_index > 0:
            for i in range(5):
                permutation_set.append(np.random.permutation(min_num))
            src_tokens_train = src_tokens_train[permutation_set[self.permut_index]]

        if self.chunk_len == -1:
            assert self.k % self.chunk_k == 0
        elif self.chunk_len + max_valid_len >= self.gpt_maxlen:
            self.chunk_len = self.gpt_maxlen - max_valid_len - 1

        print(f"NOTE: max chunk length is changed to {self.chunk_len}")
        
        if not self.truncate and not self.pad_space:
            context_src_tokens_list = [[0]]
            for i in range(len(src_tokens_train)):
                if len(src_tokens_train[i][1:]) > self.chunk_len:
                    # remove too long demonstrations
                    continue
                prev_length = len(context_src_tokens_list[-1])
                if prev_length + len(src_tokens_train[i][1:]) <= self.chunk_len:
                    context_src_tokens_list[-1] = context_src_tokens_list[-1] + src_tokens_train[i][1:]
                else:
                    context_src_tokens_list.append(src_tokens_train[i])
            
            if len(context_src_tokens_list) > 1:
                context_src_tokens_list.pop(-1)
        elif self.pad_space:
            context_src_tokens_list = [[0]]
            max_len = 0
            for i in range(len(src_tokens_train)):
                if len(src_tokens_train[i][1:]) > self.chunk_len:
                    # remove too long demonstrations
                    continue
                prev_length = len(context_src_tokens_list[-1])
                if prev_length + len(src_tokens_train[i][1:]) <= self.chunk_len:
                    context_src_tokens_list[-1] = context_src_tokens_list[-1] + src_tokens_train[i][1:]
                else:
                    max_len = max(max_len, len(context_src_tokens_list[-1]))
                    context_src_tokens_list.append(src_tokens_train[i])
            
            if len(context_src_tokens_list) > 1:
                context_src_tokens_list.pop(-1)
            
            for i in range(len(context_src_tokens_list)):
                need_len = max_len - len(context_src_tokens_list[i])
                # 1437 is the token id of space
                context_src_tokens_list[i] = [0] + need_len * [1437] + context_src_tokens_list[i][1:]
        else:
            context_src_tokens_list = [[0]]
            for i in range(len(src_tokens_train)):
                if len(src_tokens_train[i][1:]) > self.chunk_len:
                    # remove too long demonstrations
                    continue
                prev_length = len(context_src_tokens_list[-1])
                if prev_length <= self.chunk_len:
                    context_src_tokens_list[-1] = context_src_tokens_list[-1] + src_tokens_train[i][1:]
                else:
                    cur_len = len(context_src_tokens_list[-1])
                    assert cur_len >= self.chunk_len
                    context_src_tokens_list[-1] = [0] + context_src_tokens_list[-1][cur_len-self.chunk_len+1:]
                    assert len(context_src_tokens_list[-1]) == self.chunk_len
                    context_src_tokens_list.append(src_tokens_train[i])

            if len(context_src_tokens_list) == 1:
                cur_len = len(context_src_tokens_list[-1])
                context_src_tokens_list[-1] = [0] + context_src_tokens_list[-1][cur_len-self.chunk_len+1:]
            
            if len(context_src_tokens_list) > 1:
                context_src_tokens_list.pop(-1)

        print(f"NOTE: number of groups is {len(context_src_tokens_list)}, max valid len is {max_valid_len}")
        context_src_tokens_list = np.array(context_src_tokens_list)

        return src_tokens_valid, gpt_loss_mask_valid, labels, context_src_tokens_list, self.answer_set


    def tokenized_data(self, split='train'):
        src_tokens, gpt_loss_mask, labels = [], [], []
        max_len = -1
        dataset = self.dataset_train if split == 'train' else self.dataset_valid
        min_num = min(len(dataset), self.k) if split == 'train' else len(dataset)

        def encode(sentence):
            splitlines = list(filter(None, sentence.splitlines()))
            return torch.cat([self.dictionary.encode_line(self.tokenizer.encode(line), add_if_not_exist=False) for line in splitlines]).tolist()

        for i in range(min_num):
            example = dataset[i]
            if split == 'train':
                input_str, label_str, label = self.preprocess_example(example)
                if label is None:
                    continue
                if i < 2:
                    print(f"input str train is {input_str}")
                    print(f"label str train is {label_str}")

                input_str, label_str = input_str[label], label_str[label]
                input_token = encode(input_str)[0:-1]
                label_token = encode(label_str)

                src_tokens.append([0] + input_token + label_token)
                cur_length = len([0] + input_token + label_token)
                gpt_loss_mask.append([False]*cur_length)
                labels.append(label)
                max_len = max(max_len, cur_length)
            elif split == 'valid':
                input_str, label_str, label = self.preprocess_example(example)
                if label is None:
                    continue

                if label == -1:
                    self.answer_set.append(label_str)
                
                if i < 2:
                    print(f"input str valid is {input_str}")
                    print(f"label str valid is {label_str}")

                for j in range(len(input_str)):
                    sub_input_str, sub_label_str = input_str[j], label_str[j]
                    input_token = encode(sub_input_str)[0:-1]
                    label_token = encode(sub_label_str)[0:-1]

                    if label != -1:
                        src_tokens.append([0] + input_token + label_token)
                        cur_length = len([0] + input_token + label_token)
                        gpt_loss_mask.append([False]*(len(input_token)+1) + [True]*len(label_token))
                    else:
                        src_tokens.append([0] + input_token)
                        cur_length = len([0] + input_token) + 50
                        gpt_loss_mask.append([False]*(len(input_token)+1))
                    
                    labels.append(label)
                    max_len = max(max_len, cur_length)

        return np.array(src_tokens), np.array(gpt_loss_mask), np.array(labels), max_len


class WiC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'wic')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("{sentence1} {sentence2} Question: Is the word \"{word}\" used in the same way in the two sentences above? Answer:", " {answer}", ["No", "Yes"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{word}", example["word"]).replace("{sentence1}", example["sentence1"]).replace("{sentence2}", example["sentence2"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class WSC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'wsc')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("{text} In the previous sentence, does the pronoun \"{span2_text}\" refer to {span1_text}?", " {answer}", ["No", "Yes"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", example["text"]).replace("{span2_text}", example["span2_text"]).replace("{span1_text}", example["span1_text"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class MultiRC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'multirc')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

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
        answer_str = output_temp.replace("{answer}", options[example["label"]])
        options_list = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        return input_str, answer_str, options_list

# completion
class HellaSwag(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('hellaswag')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 4

    def preprocess_example(self, example):
        input_str = [example["ctx"]] * self.class_num
        answer_str = []
        for i in range(self.class_num):
            answer_str.append(' ' + example["endings"][i])
        label = int(example["label"])
        return input_str, answer_str, label

class COPA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'copa')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

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
            input_str = [text_first] * self.class_num
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
            input_str = [text_first] * self.class_num
            answer_str = [' '+example["choice1"], ' '+example["choice2"]]
        elif self.temp_index == 4:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " This happened because"
            else:
                text_first = text_first + " As a consequence,"
            input_str = [text_first] * self.class_num
            answer_str = [copa_answer(example["choice1"]), copa_answer(example["choice2"])]
        label = example["label"]
        return input_str, answer_str, label

# download manually 
class StoryCloze(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('story_cloze', '2016', data_dir="download manually")
        self.dataset_train = self.get_dataset_train(dataset['validation'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.class_num = 2

    def preprocess_example(self, example):
        input_str = example["input_sentence_1"] + ' ' + example["input_sentence_2"] + ' ' + example["input_sentence_3"] + ' ' + example["input_sentence_4"] 
        input_str = [input_str] * self.class_num
        answer_str = [' '+example["sentence_quiz1"], ' '+example["sentence_quiz2"]]
        label = int(example["answer_right_ending"]) - 1
        return input_str, answer_str, label

# Winograd tasks
class Winogrande(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('winogrande', 'winogrande_xs')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        cut_index = example["sentence"].index('_')
        text_first = example["sentence"][:cut_index]
        text_second = example["sentence"][cut_index+1:]
        input_str = [text_first+example["option1"], text_first+example["option2"]]
        answer_str = [text_second] * self.class_num
        label = int(example["answer"]) - 1
        return input_str, answer_str, label

# remove | only test split
class Winograd(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('winograd_wsc', 'wsc285')
        self.dataset_train = self.get_dataset_train(dataset['test'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.class_num = 2

    def preprocess_example(self, example):
        text_first = example["text"][:example["pronoun_loc"]]
        text_second = example["text"][example["pronoun_loc"]+len(example["pronoun"]):]
        input_str = []
        for option in example["options"]:
            input_str.append(text_first+option)
        answer_str = [text_second] * self.class_num
        label = example["label"]
        return input_str, answer_str, label


# common sense
class PIQA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('piqa')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        input_str = example["goal"]
        input_str = [input_str] * self.class_num
        answer_str = [' '+example["sol1"], ' '+example["sol2"]]
        label = example["label"]
        return input_str, answer_str, label


class OBQA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('openbookqa', 'main')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 4

    def preprocess_example(self, example):
        input_str = example["question_stem"]
        input_str = [input_str] * 4
        answer_str = []
        for i in range(4):
            answer_str.append(' '+example["choices"]["text"][i])
        
        if example["answerKey"] == 'A':
            label = 0
        elif example["answerKey"] == 'B':
            label = 1
        elif example["answerKey"] == 'C':
            label = 2
        elif example["answerKey"] == 'D':
            label = 3
        return input_str, answer_str, label


class ARCE(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('ai2_arc', 'ARC-Easy')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 4

    def preprocess_example(self, example):
        input_str = example["question"]
        input_str = [input_str] * 4
        answer_str = []
        if len(example["choices"]["text"]) != 4:
            return None, None, None

        for i in range(4):
            answer_str.append(' '+example["choices"]["text"][i])
        
        if example["answerKey"] == 'A':
            label = 0
        elif example["answerKey"] == 'B':
            label = 1
        elif example["answerKey"] == 'C':
            label = 2
        elif example["answerKey"] == 'D':
            label = 3
        else:
            label = int(example["answerKey"]) - 1
        return input_str, answer_str, label


class ARCC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('ai2_arc', 'ARC-Challenge')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 4

    def preprocess_example(self, example, split="train"):
        input_str = example["question"]
        input_str = [input_str] * 4
        answer_str = []

        if len(example["choices"]["text"]) != 4:
            return None, None, None

        for i in range(4):
            answer_str.append(' '+example["choices"]["text"][i])
        
        if example["answerKey"] == 'A':
            label = 0
        elif example["answerKey"] == 'B':
            label = 1
        elif example["answerKey"] == 'C':
            label = 2
        elif example["answerKey"] == 'D':
            label = 3
        else:
            label = int(example["answerKey"]) - 1
        return input_str, answer_str, label

class RACEm(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('race', "middle")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 4
    
    def templates_set_without_newline(self):
        return [
            ("Passage: {article} Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{article}", example["article"]).replace("{question}", example["question"])
        input_str = [input_str] * self.class_num

        if len(example["options"]) != 4:
            return None, None, None

        answer_str = [output_temp.replace("{answer}", answer) for answer in example["options"]]

        if example["answer"] == 'A':
            label = 0
        elif example["answer"] == 'B':
            label = 1
        elif example["answer"] == 'C':
            label = 2
        elif example["answer"] == 'D':
            label = 3
        else:
            label = int(example["answer"]) - 1
        return input_str, answer_str, label

class RACEh(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('race', "high")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 4
    
    def templates_set_without_newline(self):
        return [
            ("Passage: {article} Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{article}", example["article"]).replace("{question}", example["question"])
        input_str = [input_str] * self.class_num

        if len(example["options"]) != 4:
            return None, None, None

        answer_str = [output_temp.replace("{answer}", answer) for answer in example["options"]]

        if example["answer"] == 'A':
            label = 0
        elif example["answer"] == 'B':
            label = 1
        elif example["answer"] == 'C':
            label = 2
        elif example["answer"] == 'D':
            label = 3
        else:
            label = int(example["answer"]) - 1
        return input_str, answer_str, label

# glue
class SST2(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('glue', 'sst2')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2
        self.punct = tuple(string.punctuation)

    def templates_set_without_newline(self):
        return [
            ("Sentence: {sentence} Label:", " {answer}", ["Negative", "Positive"])
        ]

    def preprocess_example(self, example, split="train"):
        sentence = example["sentence"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{sentence}", sentence)
        # input_str = random.randint(0, 10) * " " + input_str

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label


class AGNews(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('ag_news')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 4

    def templates_set_without_newline(self):
        return [
            ("Classify the news articles into the categories of World, Sports, Business, and Technology. Article: {text} Answer:", " {answer}", ["World", "Sports", "Business", "Technology"]),
            ("News: {text} Type:", " {answer}", ["World", "Sports", "Business", "Technology"]),
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label


class SST5(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('SetFit/sst5')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 5

    def templates_set_without_newline(self):
        return [
            ("Sentence: {text} Label:", " {answer}", ["terrible", "bad", "neutral", "good", "great"])
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class TREC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('trec')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 6
        self.punct = tuple(string.punctuation)

    def templates_set_without_newline(self):
        return [
            ("Question: {text} Type:", " {answer}", ["Description", "Entity", "Expression", "Person", "Number", "Location"])
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label-coarse"]
        return input_str, answer_str, label

class RTE(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("glue", "rte")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Passage: {sentence1} Question: {sentence2} Yes or No? Answer:", " {answer}", ["Yes", "No"])
        ]

    def preprocess_example(self, example, split="train"):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{sentence1}", sentence1).replace("{sentence2}", sentence2)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class IMDB(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("imdb")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Review: {text} Is the review positive or negative?", " {answer}", ["Negative", "Positive"])
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class Subj(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("SetFit/subj")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Input: {text} Type:", " {answer}", ["objective", "subjective"])
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class DBPedia(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("dbpedia_14")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 14

    def templates_set_without_newline(self):
        return [
            ("Input: {content} Type:", " {answer}", ["company", "school", "artist", "athlete", "politics", "transportation", "building", "nature", "village", "animal", "plant", "album", "film", "book"])
        ]

    def preprocess_example(self, example, split="train"):
        content = example["content"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{content}", content)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class MR(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset("rotten_tomatoes")
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Review: {text} Sentiment:", " {answer}", ["Negative", "Positive"])
        ]

    def preprocess_example(self, example, split="train"):
        text = example["text"]

        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", text)

        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

class CB(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'cb')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 3

    def templates_set_without_newline(self):
        return [
            ("{premise} Question: {hypothesis}. True, False, or Neither? Answer:", " {answer}", ["True", "False", "Neither"]),
            ("{premise} Based on the paragraph above can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", " Answer: {answer}.", ["Yes", "No", "Maybe"]),
            ("{premise} Can we infer the following? {hypothesis}.", " {answer}", ["Yes", "No", "Maybe"]),
            ("Read the following paragraph and determine if the hypothesis is true: {premise} Hypothesis: {hypothesis}.", " {answer}", ["Yes", "No", "Maybe"]),
            ("Can we draw the following hypothesis from the context? Context: {premise} Hypothesis: {hypothesis}. Answer:", " {answer}", ["Yes", "No", "Maybe"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{premise}", example["premise"]).replace("{hypothesis}", example["hypothesis"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

# qa
class BoolQ(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'boolq')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Passage: {passage} Question: {question}? Answer:", " {answer}", ["No", "Yes"]),
            ("Passage: {passage} After reading this passage, I have a question: {question}? True or False?", " {answer}", ["False", "True"]),
            ("Text: {passage} Question: {question}? Answer:", " {answer}", ["No", "Yes"]),
            ("{passage} Based on the above text, what's the best answer to this question: {question}?", " {answer}", ["No", "Yes"]),
            ("Based on the following passage, {question}? {passage} Please answer yes or no.", " {answer}", ["No", "Yes"]),
            ("Exercise: read the text and answer the question by True or False. Text: {passage} Question: {question}?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example, split="train"):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{passage}", example["passage"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label

# open-ended QA
class NQ(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('nq_open')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Question: {question}? Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", answer) for answer in example["answer"]]
        return input_str, answer_str, -1

class WebQS(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('web_questions')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", answer) for answer in example["answers"]]
        return input_str, answer_str, -1

class TriviaQA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('trivia_qa', 'rc.nocontext')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", answer) for answer in example["answer"]["aliases"]]
        return input_str, answer_str, -1

# extractive QA
class SQuADv2(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('squad_v2')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Passage: {context} Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{context}", example["context"])
        input_str = [input_str] * self.class_num

        if len(example["answers"]["text"]) > 0:
            answer_str = [output_temp.replace("{answer}", answer) for answer in example["answers"]["text"]]
        else:
            answer_str = [output_temp.replace("{answer}", "none")]
        return input_str, answer_str, -1

class SQuAD(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('squad')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Passage: {context} Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{context}", example["context"])
        input_str = [input_str] * self.class_num

        if len(example["answers"]["text"]) > 0:
            answer_str = [output_temp.replace("{answer}", answer) for answer in example["answers"]["text"]]
        else:
            answer_str = [output_temp.replace("{answer}", "none")]
        return input_str, answer_str, -1


class COQA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('Zaid/coqa_expanded')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 1

    def templates_set_without_newline(self):
        return [
            ("Passage: {story} Question: {question} Answer:", " {answer}")
        ]

    def preprocess_example(self, example):
        input_temp, output_temp = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{story}", example["story"])
        input_str = [input_str] * self.class_num

        if len(example["answer"]["input_text"]) > 0:
            answer_str = [output_temp.replace("{answer}", example["answer"]["input_text"])]
        else:
            answer_str = [output_temp.replace("{answer}", "none")]
        return input_str, answer_str, -1
