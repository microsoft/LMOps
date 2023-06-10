from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import pandas as pd
import json
import pandas as pd
from datasets import Dataset
import json
import re
from src.utils.dataset_utils import pad2sameLen
from DPR.dpr.utils.tasks import task_map

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


class ScorerDatasetReader(torch.utils.data.Dataset):
    def __init__(
        self,
        example_file,
        model_name,
        task_name,
        prompt_pool_path=None,
        cache_dir=None,
        max_length=2048,
    ) -> None:
        self.task = task_map.cls_dic[task_name]()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, model_max_length=max_length
        )
        if self.task.class_num == 1:  # text completion question
            self.tokenizer.padding_side = "left"

        # prompt_pool
        with open(prompt_pool_path, "r", encoding="utf-8") as f:
            prompt_pool = json.load(f)
        self.prompt_pool = list(enumerate(prompt_pool))
        

        # task_data
        with open(example_file) as f1:
            self.task_data = json.load(f1)

        def get_instance(entry):
            examples = entry.pop("ctxs")
            for exp in examples:
                exp.update(self.prompt_pool[exp["id"]][1])
                for key, val in entry.items():
                    exp[f"test_{key}"] = val
            yield from examples

        def get_dataset(data):
            for entry in data:
                yield from get_instance(entry)

        df = pd.DataFrame(list(get_dataset(self.task_data)))
        self.dataset = Dataset.from_pandas(df)

    def shard(self, accelerator):
        self.dataset = self.dataset.shard(
            num_shards=accelerator.num_processes, index=accelerator.process_index
        )

    def __getitem__(self, index):
        if self.task.class_num == 1: # text completion question
            return self.text_to_instance_completion(self.dataset[index])
        else:
            return self.text_to_instance_choice(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def get_fields(self, entry):
        example = {}
        for key, val in entry.items():
            if key.startswith("test_"):
                example[key[len("test_") :]] = val

        test_input_strs = self.task.get_input_strs(example)
        question = self.task.get_question(entry)
        answer = self.task.get_answer(entry)
        demonstration = f'{question}{answer}'
        test_questions = [demonstration + " \n " + input for input in test_input_strs]
        test_answer_strs = self.task.get_answers(example)
        test_label = self.task.get_label(example)
        return test_questions, test_answer_strs, test_label

    def text_to_instance_choice(self, entry):
        """
        multiple-choice question
        """
        test_questions, test_answers, test_label = self.get_fields(entry)  

        input_ids_list = []
        input_atten_mask_list = []
        input_loss_mask_list = []
        for i in range(len(test_questions)):
            enc_text = remove_double_space(test_questions[i] + test_answers[i])
            enc_answer = remove_double_space(test_answers[i])
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized_answer = self.tokenizer.encode_plus(
                enc_answer,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )


            answer_mask = tokenized_answer.attention_mask.squeeze()
            if len(answer_mask.shape) == 0:
                answer_mask = torch.tensor([1]).to(answer_mask)

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()
            input_loss_mask = torch.nn.functional.pad(
                answer_mask, (input_ids.shape[-1] - answer_mask.shape[-1], 0)
            )

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            input_loss_mask_list.append(input_loss_mask)

        return {
            "input_ids": pad2sameLen(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id
            ),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0),
            "input_loss_mask": pad2sameLen(input_loss_mask_list, pad_idx=0),
            "labels": torch.tensor([test_label]),
            "metadata": entry,
        }

    def text_to_instance_completion(self, entry: Dict[str, Any]):
        """
        text completion question
        """
        test_questions, _, test_label = self.get_fields(entry)

        input_ids_list = []
        input_atten_mask_list = []
        for i in range(len(test_questions)): # len(test_questions) = 1 for completion question
            enc_text = remove_double_space(test_questions[i]).strip() 
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)

        entry["temp_label"] = test_label  # pass label for the next step
        return {
            "input_ids": pad2sameLen(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "input_atten_mask": pad2sameLen(
                input_atten_mask_list, pad_idx=0, left_pad=True
            ),
            "metadata": entry,
        }
