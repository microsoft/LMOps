'''
for knn search
'''
from datasets import load_dataset
from typing import Any, Dict, Iterable
import torch
import pandas as pd
import tqdm

class IndexerDatasetReader(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data) -> None:
        self.tokenizer = tokenizer
        self.dataset=data
                 
    def __getitem__(self, index):
        return self.text_to_instance(self.dataset[index],index=index)

    def __len__(self):
        return len(self.dataset)

    def text_to_instance(self, entry: Dict[str, Any],index=-1):
        enc_text = entry['instruction']
        tokenized_inputs = self.tokenizer.encode_plus(enc_text,truncation=True,return_tensors='pt')
        return {
                        'input_ids': tokenized_inputs.input_ids.squeeze(),
                        'attention_mask': tokenized_inputs.attention_mask.squeeze(),
                        "metadata":{"id":index}
                        
                    }

        
