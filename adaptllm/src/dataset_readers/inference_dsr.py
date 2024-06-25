from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import more_itertools
from src.utils.dataset_utils import pad2sameLen
from src.dataset_readers.task import task_map

class InferenceDatasetReader(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
        task_name,
        n_tokens=2048,
        cache_dir=None,
        max_length=2048,
        generate_max_len=100,
        add_bos_token=False
    ) -> None:
        self.task = task_map.cls_dic[task_name]()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            model_max_length=max_length,
            truncation_side="left",
            trust_remote_code=True
        )

        if self.task.class_num == 1:  # text completion
            self.tokenizer.padding_side = "left"

        self.test_split = self.task.get_dataset(cache_dir=cache_dir)
        self.n_tokens_in_prompt = n_tokens
        self.generate_max_len = generate_max_len
        self.num_processes = 1
        self.process_index = 0
        self.add_bos_token = add_bos_token

    def __getitem__(self, index):
        if self.task.class_num == 1:
            # text completion
            return self.text_to_instance_completion(self.test_split[index])
        else: 
            # multiple choice
            return self.text_to_instance_choice(self.test_split[index])

    def __len__(self):
        return len(self.test_split)

    def shard(self, accelerator):
        self.num_processes = accelerator.num_processes
        self.process_index = accelerator.process_index
        self.test_split = list(more_itertools.distribute(accelerator.num_processes, self.test_split)[accelerator.process_index])

    def get_fields(self, entry):
        answers = self.task.get_answers(entry)
        questions = self.task.get_input_strs(entry, class_num=len(answers))
        label = self.task.get_label(entry)          
        return questions, answers, label

    def text_to_instance_choice(self, entry: Dict[str, Any]):
        """
        multiple-choice question
        """
        questions, answers, label = self.get_fields(entry)
        input_ids_list = []
        input_atten_mask_list = []
        loss_mask_list = []

        entry["infer_q"] = []
        entry["infer_a"] = []
        for i in range(len(questions)):
            qa = questions[i] + answers[i]
            q = questions[i]
            tokenized_qa = self.tokenizer.encode_plus(qa, truncation=False, return_tensors="pt", add_special_tokens=self.add_bos_token)
            tokenized_q = self.tokenizer.encode_plus(q, truncation=False, return_tensors="pt", add_special_tokens=self.add_bos_token)
            q_mask = tokenized_q.attention_mask.squeeze()
            if len(q_mask.shape) == 0:
                q_mask = torch.tensor([1]).to(q_mask)

            input_ids = tokenized_qa.input_ids.squeeze()
            input_atten_mask = tokenized_qa.attention_mask.squeeze()

            loss_mask = torch.tensor([0] * q_mask.shape[-1] + [1] * (input_ids.shape[-1] - q_mask.shape[-1])).to(input_ids)

            a = answers[i]
            
            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            loss_mask_list.append(loss_mask)
            entry["infer_q"].append(q)
            entry["infer_a"].append(a)

        entry["label"] = label
        return {
            "input_ids": pad2sameLen(input_ids_list, pad_idx=self.tokenizer.pad_token_id),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0),
            "loss_mask": pad2sameLen(loss_mask_list, pad_idx=0),
            "labels": torch.tensor([label]),
            "metadata": entry,
        }

    def text_to_instance_completion(self, entry: Dict[str, Any]):
        """
        text completion question
        """
        questions, answers, label = self.get_fields(entry)
        input_ids_list = []
        input_atten_mask_list = []

        entry["infer_q"] = []
        entry["infer_a"] = []
        for i in range(len(questions)):
            q = questions[i]
            tokenized_q = self.tokenizer.encode_plus(
                        q,
                        truncation=True,  # truncate from left for long inputs
                        max_length=self.n_tokens_in_prompt - self.generate_max_len,
                        return_tensors="pt",
                        add_special_tokens=self.add_bos_token,
                        )

            input_ids = tokenized_q.input_ids.squeeze()
            input_atten_mask = tokenized_q.attention_mask.squeeze()

            if len(input_ids.shape) == 0:
                input_ids = input_ids.unsqueeze(0)
                input_atten_mask = input_atten_mask.unsqueeze(0)

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)

            a = answers[i]
            entry["infer_q"].append(q)
            entry["infer_a"].append(a)
        entry["label"] = label
        return {
            "input_ids": pad2sameLen(input_ids_list, pad_idx=self.tokenizer.pad_token_id, left_pad=True),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0, left_pad=True),
            "metadata": entry,
        }