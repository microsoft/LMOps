import json
import os
import re
import random
import numpy as np
from transformers import AutoTokenizer
from functools import partial
from datasets import load_dataset, load_metric, DatasetDict, concatenate_datasets
from typing import Tuple, Dict


def get_dataset(task, tokenizer: AutoTokenizer, save_dir: str, max_source_length: int) -> Tuple[DatasetDict, DatasetDict, 'BaseProcessor']:
    task = task.lower()
    if task == 'gsm8k':
        train_dataset = load_dataset('gsm8k', split='train', name='main')
        test_dataset = load_dataset('gsm8k', split='test', name='main')
        processor = GSM8KProcessor(tokenizer, save_dir, max_source_length)
    elif task == 'svamp':
        train_dataset = load_dataset('ChilleD/SVAMP', split='train')
        test_dataset = load_dataset('ChilleD/SVAMP', split='test')
        processor = SVAMPProcessor(tokenizer, save_dir, max_source_length)
    elif task == 'mathqa':
        train_dataset = load_dataset('math_qa', split='train')
        test_dataset = load_dataset('math_qa', split='test')
        processor = MathqaProcessor(tokenizer, save_dir, max_source_length)
    elif task == 'mmqa':
        train_dataset = load_dataset('meta-math/MetaMathQA', split='train')
        test_dataset = None
        processor = MetaMathQAProcessor(tokenizer, save_dir, max_source_length)
    elif task == 'hs':
        train_dataset = load_dataset('Rowan/hellaswag', split='train').rename_column('label', 'label_text')
        test_dataset = load_dataset('Rowan/hellaswag', split='validation').rename_column('label', 'label_text')
        processor = HSProcessor(tokenizer, save_dir, max_source_length)
    else:
        raise NotImplementedError(task)
    return train_dataset, test_dataset, processor

class BaseProcessor:
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str, max_source_length: int):
        self.tokenizer = tokenizer
        
        self.best_result = 0.0
        self.drop_bos = True
        self.answer_prompt = ''
        self.max_source_length = max_source_length
        self.max_input_length = 0
        self.save_dir = save_dir
        self.eval_epoch = 0
        self.test_dataset = None
        self.process_train = partial(self.my_process, train_flag=True)
        self.process_test = partial(self.my_process, train_flag=False)

    @staticmethod
    def filter_long_examples(example: Dict,) -> bool:
        return len(example['input_ids']) <= example['my_max_length']

    @staticmethod
    def post_process(data: DatasetDict, final_num: int = -1):
        print("Info: before post_process, total data:", len(data))
        data = data.filter(BaseProcessor.filter_long_examples)
        if final_num > 0 and len(data) > final_num:
            data = data.train_test_split(test_size=final_num, shuffle=True)['test']
        print("Info: after post_process, total data:", len(data))
        return data

    def process(self, data: dict, inputs: str, labels: str, train_flag: bool = True):
        data['input_text'] = inputs
        data['label_text'] = labels
        inputs = self.tokenizer(inputs)
        labels = self.tokenizer(labels)
        if self.drop_bos:
            labels['input_ids'] = labels['input_ids'][1:]
        if train_flag:
            data['input_ids'] = inputs['input_ids'] + labels['input_ids'] + [self.tokenizer.eos_token_id]
            data['labels'] = [-100] * len(inputs['input_ids']) + labels['input_ids'] + [self.tokenizer.eos_token_id]
            assert (len(data['input_ids']) == len(data['labels'])), "Error: input_ids and labels have different length!"
            data['attention_mask'] = [1] * len(data['input_ids'])
        else:
            data['input_ids'] = inputs['input_ids']
            data['labels'] = inputs['input_ids']
            
        self.max_input_length = max(self.max_input_length, len(data['input_ids']))
        data['my_max_length'] = self.max_source_length

        return data

    def my_process(self, data, train_flag: bool):
        raise NotImplementedError

    def compute_metrics(self, eval_preds):
        tokenizer = self.tokenizer
        self.eval_epoch += 1

        preds, raw_labels = eval_preds
        num_correct, total_problem = 0, len(preds)

        for i in range(len(preds)):
            preds[i] = np.where(preds[i] != -100, preds[i], tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [x['label_text'] for x in self.test_dataset]
        if len(preds) > len(labels):
            print(f"Warning: preds and labels have different length! preds num is {len(preds)}, labels num is {len(labels)}")
            preds = preds[:len(labels)]

        assert len(preds) == len(labels), f"Error: preds and labels have different length! {len(preds)} vs {len(labels)}"
        temp_ans = []

        for p, l in zip(preds, labels):
            p = p.strip()
            l = l.strip()
            temp_ans.append({'pred': p, 'label': l})
            if self.answer_prompt != "":
                if self.answer_prompt in p:
                    p = p.split(self.answer_prompt)[1].strip()
                else:
                    p = ""
                if self.answer_prompt in l:
                    l = l.split(self.answer_prompt)[1].strip()
                else:
                    print("Error: no answer in label!", l)
            if p == l:
                num_correct += 1

        with open(os.path.join(self.save_dir, f'eval_result_{self.eval_epoch}.json'), 'w') as f:
            json.dump({"ans": temp_ans}, f, indent=4)
        result = round(num_correct / total_problem * 100, 2)
        self.best_result = max(self.best_result, result)
        print(f'Best Exactly Match: {self.best_result}')

        return {'Accuracy': result, 'Best Accuracy': self.best_result}

    @staticmethod
    def sort_by_length(dataset: DatasetDict):
        dataset = dataset.map(lambda x: {'length': len(x['input_ids'])})
        dataset = dataset.sort("length", reverse=True)
        return dataset


class GSM8KProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str = "", max_source_length: int = 600):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"

    @staticmethod
    def clean(content):
        pattern = '<<.+>>'
        result = re.findall(pattern, content)
        for t in result:
            content = content.replace(t, '')
        content = content.replace('\n', ' ')
        return content

    def my_process(self, data, train_flag: bool):
        prompt = data['question']
        ans = GSM8KProcessor.clean(data['answer'])
        ans = ans.replace('####', self.answer_prompt) + '.'
        return self.process(data, prompt, ans, train_flag)


class SVAMPProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str = "", max_source_length: int = 600):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"

    def my_process(self, data, train_flag: bool):
        prompt = data['Body'] + data['Question']
        
        ans = f"{data['Equation']}. {self.answer_prompt} {data['Answer']}."
        return self.process(data, prompt, ans, train_flag)


class AquaProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str = "", max_source_length: int = 600):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"

    def my_process(self, data, train_flag: bool):
        prompt = f"{data['question']}\nThere are {len(data['options'])} options: "
        prompt += ', '.join(data['options'])
        prompt += '\nPlease answer the correct option.'
        ans = f"{data['rationale']} {self.answer_prompt} {data['correct']}."
        return self.process(data, prompt, ans, train_flag)


class MathqaProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str, max_source_length: int):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"

    def my_process(self, data, train_flag: bool):
        prompt = f"{data['Problem']}\nThere are {len(data['options'].split(')'))-1} options: {data['options']}"
        prompt += '\nPlease answer the correct option.'
        ans = f"{data['Rationale']}. {self.answer_prompt} {data['correct']}."
        return self.process(data, prompt, ans, train_flag)

class MetaMathQAProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str, max_source_length: int):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"
        self.old_answer_prompt = "The answer is:"

    def my_process(self, data, train_flag: bool):
        prompt = f"{data['query']}"
        ans = data['response'].replace(self.old_answer_prompt, self.answer_prompt) + '.'
        return self.process(data, prompt, ans, train_flag)


class HSProcessor(BaseProcessor):
    def __init__(self, tokenizer: AutoTokenizer, save_dir: str, max_source_length: int):
        super().__init__(tokenizer, save_dir, max_source_length)
        self.answer_prompt = "The answer is"

    def my_process(self, data, train_flag: bool):
        prompt = f"Here is a sentence to be continued: \"{data['ctx']}\"\nThere are 4 options as an ending:\n"
        sens = []
        for i in range(len(data['endings'])):
            sens.append(f"{i}: {data['endings'][i]}")
        sentences = "\n".join(sens)
        prompt += sentences + f"\nPlease select the correct option and answer with a number between 0 and 3."
        ans = f"{self.answer_prompt} {data['label_text']}."
        return self.process(data, prompt, ans, train_flag)


def add_eos_token(sentences: list, eos_token: str) -> list:
    if eos_token is None:
        return sentences
    else:
        return [f"{s} {eos_token}" for s in sentences]


def test(name='aqua'):
    print(f"loading dataset {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/aiscuser/models/Llama-2-7b-hf")
    train_dataset, test_dataset, processor = get_dataset(name, tokenizer, "./eval_result/", 4096)
    print(train_dataset, test_dataset)
    train_dataset = train_dataset.select(range(100))
    train_dataset = train_dataset.map(processor.process_train, load_from_cache_file=False)
    print(train_dataset, test_dataset)
    demo = "Sample:\n"
    for i in range(3):
        demo += f"{train_dataset[i]['input_text']}\n{train_dataset[i]['label_text']}\n\n"
    print(f'\n{demo}')


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2
    print(sys.argv)
    test(sys.argv[1])
