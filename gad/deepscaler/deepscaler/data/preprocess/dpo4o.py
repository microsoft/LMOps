import json
import random
import pandas as pd
import re
from tqdm import tqdm


# "remove", "keep", "replace"
think_process_strategy = "remove"
think_process_strategy = "replace"


def main():
  src_fn = "/mnt/mnhotzc/data/po/4o_dpo_167k.jsonl"
  # trg_fn = "/mnt/mnhotzc/data/po/po4o_no_think_train.parquet"
  trg_fn = "/mnt/mnhotzc/data/po/po4o_replace_think_train.parquet"
  trg_df = []

  orig_response_len_sum = 0
  nothink_response_len_sum = 0
  count = 0


  def _remove_think(text):
    # pos = text.rfind(r"</think>")
    pattern = r"</think>"
    pos = list(re.finditer(pattern, text))
    if len(pos) == 0:
      return text
    pos = pos[-1].start()
    text = text[pos+len(r"</think>"):]
    return text

  def _replace_think(text):
    text = text.replace(r"<think>", "")
    text = text.replace(r"</think>", "")
    return text

  with open(src_fn, "r") as f:
    for idx, line in tqdm(enumerate(f), total=167000):
      example = json.loads(line)

      if think_process_strategy == "remove":
        orig_response_len_sum += len(example["chosen"])
        chosen = _remove_think(example["chosen"])
        nothink_response_len_sum += len(chosen)

        orig_response_len_sum += len(example["rejected"])
        rejected = _remove_think(example["rejected"])
        nothink_response_len_sum += len(rejected)
        count += 2
      
      elif think_process_strategy == "keep":
        chosen = example["chosen"]
        rejected = example["rejected"]
      elif think_process_strategy == "replace":
        orig_response_len_sum += len(example["chosen"])
        chosen = _replace_think(example["chosen"])
        nothink_response_len_sum += len(chosen)

        orig_response_len_sum += len(example["rejected"])
        rejected = _replace_think(example["rejected"])
        nothink_response_len_sum += len(rejected)
        count += 2
      else:
        raise ValueError("Invalid think_process_strategy")

      new_dict = {
        "data_source": "skywork_raw_v2",
        "question": example["prompt"],
        "response1": chosen,
        "response2": rejected,
        "answer": "1",
      "extra_info": {'index': idx, 'split': 'train'}
      }
      if idx ==0:
        print(new_dict)
      trg_df.append(new_dict)
  
  #shuffle the data
  random.shuffle(trg_df)
  print(f"length of trg_df: {len(trg_df)}")
  print(f"orig_response_len_avg: {orig_response_len_sum/count}, nothink_response_len_avg: {nothink_response_len_sum/count}")
  trg_df = pd.DataFrame(trg_df)
  trg_df.to_parquet(trg_fn)
  print(f"saved to {trg_fn}")

if __name__ == '__main__':
  main()