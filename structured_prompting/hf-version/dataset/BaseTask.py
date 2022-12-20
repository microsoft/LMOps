import random

import torch
from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, max_data_num=None, temp_index=0, demo=""):
        super().__init__()
        self.temp_index = temp_index
        self.examples = []
        self.max_data_num = max_data_num
        self.demo = demo
        self.templates = self.templates_set_without_newline()
        
    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def preprocess_dataset(self):
        self.label_count = [0 for _ in range(self.class_num)]
        for example in self.dataset:
            example = self.preprocess_example(example)
            if example[0] is None:
                continue

            self.label_count[example[2]] += 1
            self.examples.append(example)
 
        if self.max_data_num is not None and self.max_data_num < len(self.examples): # truncate dataset
            random.seed(1)
            self.examples = random.sample(self.examples, self.max_data_num)
            # num_each_label_float = torch.Tensor(self.label_count) / len(self.examples) * self.max_data_num
            # num_each_label = num_each_label_float.long()
            # _, indices = torch.sort(num_each_label_float - num_each_label, descending=True)
            # for index in indices:
            #     if num_each_label.sum().item() >= self.max_data_num:
            #         break
            #     num_each_label[index] += 1

            # num_each_label_count = [0 for _ in range(self.class_num)]
            # new_examples = []
            # for example in self.examples:
            #     label = example[2]
            #     if num_each_label_count[label] < num_each_label[label]:
            #         new_examples.append(example)
            #         num_each_label_count[label] += 1
            
            # self.examples = new_examples

    def get_demo_from_indices(self, indices):
        demo_str = ""
        if isinstance(indices, int):
            indices = [indices]
            
        for index in indices:
            input_str, output_str, label = self.examples[index]
            demo_str += input_str + output_str[label] + " \n "

        return demo_str

    def get_max_length(self, tokenizer):
        return max(len(tokenizer(
                [input_str +" " + candidate_str for candidate_str in output_str],
                padding=True
            ).input_ids[0]) for input_str, output_str, _ in self.examples)

    def get_chunk(self, tokenizer, max_length, strategy="truncate", indices=None, shot=None, chunk_num=None):
        num_examples = 0
        if indices is None:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)

        if shot is not None:
            indices = random.sample(indices, shot)
            
        demo_encoding_batch = []
        demo_encoding = []
        attention_mask_batch = []
        for index in indices:
            if chunk_num is not None and len(demo_encoding_batch) >= chunk_num:
                break

            demo = self.get_demo_from_indices(index)
            demo_input_ids = tokenizer(demo).input_ids
            num_examples += 1
            if len(demo_encoding) + len(demo_input_ids) <= max_length:
                demo_encoding += demo_input_ids
            else:
                if strategy == "padding":
                    demo_encoding_batch.append([0] * (max_length - len(demo_encoding)) + demo_encoding)
                    attention_mask_batch.append([0] * (max_length - len(demo_encoding)) + [1] * len(demo_encoding))
                    demo_encoding = demo_input_ids
                elif strategy == "truncate":
                    demo_encoding_batch.append((demo_encoding + demo_input_ids)[-max_length:])
                    attention_mask_batch.append([1] * max_length)
                    demo_encoding = []
                elif strategy == "blank":
                    demo_encoding_batch.append(tokenizer(" ").input_ids * (max_length - len(demo_encoding)) + demo_encoding)
                    attention_mask_batch.append([1] * max_length)
                    demo_encoding = demo_input_ids
                else:
                    demo_encoding_batch.append(demo_encoding + [0] * (max_length - len(demo_encoding)) )
                    attention_mask_batch.append([1] * len(demo_encoding) + [0] * (max_length - len(demo_encoding)))
                    demo_encoding = demo_input_ids

        if len(demo_encoding_batch) == 0: # doesn't need chunk!
            demo_encoding_batch.append(demo_encoding)
            attention_mask_batch.append([1] * len(demo_encoding))

        return demo_encoding_batch, attention_mask_batch, num_examples
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return self.demo + input_str, output_str, label

    def __iter__(self):
        for input_str, output_str, label in self.examples:
            yield self.demo + input_str, output_str, label


