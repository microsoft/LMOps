import json
import random
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import datasets
from tqdm import tqdm


def main(split: str = "train"):
  # Load dataset
  src_ds = load_dataset("TIGER-Lab/WebInstruct-verified")[split]
  trg_df = []
  trg_fn = f"./webinst_rlvr_{split}.parquet"

  for idx, data in tqdm(enumerate(src_ds)):
    question = data["question"] + " Let's think step by step and output the final answer within \\boxed{}."
    answer = data["answer"]
    idx = data["id"]
    new_dict = {
      "data_source": "",
      "prompt": [{
          "role": "user",
          "content": question
      }],
      "ability": "general",
      "reward_model": {
          "style": "rule",
          "ground_truth": answer
      },
      "extra_info": {
          'split': split,
          'index': idx
      }
    }
    trg_df.append(new_dict)
  trg_df = pd.DataFrame(trg_df)
  trg_df.to_parquet(trg_fn)
  print(f"processed {len(trg_df)} examples and saved to {trg_fn}")


if __name__ == '__main__':
  main(split="train")
  main(split="test")
  # main(split="validation")
  

    