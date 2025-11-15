import json
import random
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import datasets

# # Load dataset
# ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

# # Get dataset size
# total_size = len(ds["train"])
# train_size = 75000
# test_size = total_size - train_size

# # Split dataset
# train_test_split = ds["train"].train_test_split(train_size=train_size, test_size=test_size, shuffle=True, seed=42)

# # Combine into new dataset
# new_ds = DatasetDict({
#     "train": train_test_split["train"],
#     "test": train_test_split["test"]
# })

# # Save the new dataset to disk
# new_ds.save_to_disk("/mnt/blob/skywork-data/skywork_raw")

def transform_to_skywork_raw_v2(src_fn, trg_fn):
  src_df = pd.read_parquet(src_fn)
  suffix_instruction = " Let's analyze this step by step and decide which solution is better, and then answer \\boxed{Solution 1} or \\boxed{Solution 2}."
  trg_df = []

  for idx in range(len(src_df)):
    row_dict = src_df.iloc[idx].to_dict()
    assert len(row_dict["prompt"]) == 1
    prompt = row_dict["prompt"][0]["content"]
    question, prompt = prompt.split("\n\nChoose the better answer from the following two responses:\nSolution 1: ")
    try:
      response1, response2 = prompt.split("\nSolution 2: ")
    except:
      print(f"Error in idx {idx} with prompt: {prompt}")
      raise RuntimeError
    pos = response2.find(suffix_instruction)
    assert pos != -1
    response2 = response2[:pos]
    new_dict = {
      "data_source": "skywork_raw_v2",
      "question": question,
      "response1": response1,
      "response2": response2,
      "answer": row_dict["reward_model"]["ground_truth"],
      "extra_info": row_dict["extra_info"],
    }
    print(new_dict)
    trg_df.append(new_dict)
  trg_df = pd.DataFrame(trg_df)
  trg_df.to_parquet(trg_fn)


def transform_to_skywork_raw_v2_from_raw_v1(src_fn, trg_fn):
  trg_df = []

  src_ds = datasets.load_from_disk(dataset_path=src_fn)["train"]

  # for idx in range(len(src_ds)):
  for idx, example in enumerate(src_ds):
    # example = src_ds[idx]
    print(example)
    question = example["chosen"][0]["content"]
    new_dict = {
      "data_source": "skywork_raw_v2",
      "question": question,
      "response1": " ".join([turn["content"] for turn in example["chosen"] if turn["role"] == "assistant"]),
      "response2": " ".join([turn["content"] for turn in example["rejected"] if turn["role"] == "assistant"]),
      "answer": "1",
      "extra_info": {'index': idx, 'split': 'train'}
    }
    print(new_dict)
    trg_df.append(new_dict)
  trg_df = pd.DataFrame(trg_df)
  trg_df.to_parquet(trg_fn)


if __name__ == '__main__':
  src_fn = "/mnt/jiaxin/skywork-data//skywork_raw"
  trg_fn = "/mnt/jiaxin/skywork-data/skywork_raw_v2/train.parquet"
  transform_to_skywork_raw_v2_from_raw_v1(src_fn, trg_fn)
  # src_fn = "/mnt/jiaxin/skywork-data/skywork.parquet"
  # trg_fn = "/mnt/jiaxin/skywork-data/skywork_raw_v2/test.parquet"
  # transform_to_skywork_raw_v2(src_fn, trg_fn)