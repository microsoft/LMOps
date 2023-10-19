"""
python dl_data.py > train.jsonl
python dl_data.py > test.jsonl
"""


import json
from datasets import load_dataset

# https://huggingface.co/datasets/liar
dataset = load_dataset("liar")

train = {}
for row in dataset['test']:
    if row['label'] in {0, 3}: # true, barely true
        ex = {
            'label': 1 if row['label'] == 0 else 0,
            'text': f'Statement: {row["statement"]}\nJob title: {row["job_title"]}\nState: {row["state_info"]}\nParty: {row["party_affiliation"]}\nContext: {row["context"]}'
        }
        print(json.dumps(ex))

# with open('train.json', 'w') as f:
    

# print(dataset)
quit()