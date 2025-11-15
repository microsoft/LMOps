import json
import random
import pandas as pd
import re
from tqdm import tqdm


# "remove", "keep", "replace"
think_process_strategy = "remove"
# think_process_strategy = "replace"


def main():
  src_fn = "/mnt/mnhotzc/data/po/4o_dpo_167k.jsonl"
  # trg_fn = "/mnt/mnhotzc/data/po/po4o_no_think_train.parquet"
  # trg_fn = "/mnt/mnhotzc/data/po/po4o_replace_think_train.parquet"
  trg_rm80k_fn = "/mnt/mnhotzc/data/po/po4o_rmtk_reward80k_train.parquet"
  trg_rm80k_df = []
  trg_val2k_fn = "/mnt/mnhotzc/data/po/po4o_rmtk_valid2k.parquet"
  trg_val2k_df = []
  trg_dpo85k_fn = "/mnt/mnhotzc/data/po/po4o_rmtk_dpo85k_train.parquet"
  trg_dpo85k_df = []
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


  # trg_df = pd.DataFrame(trg_df)
  # trg_df.to_parquet(trg_fn)
  # print(f"saved to {trg_fn}")

  trg_reward80k_df = trg_df[:80000]
  trg_reward80k_df = pd.DataFrame(trg_reward80k_df)
  trg_reward80k_df.to_parquet(trg_rm80k_fn)
  print(f"saved to {trg_rm80k_fn}")

  trg_val2k_df = trg_df[80000:82000]
  trg_val2k_df = pd.DataFrame(trg_val2k_df)
  trg_val2k_df.to_parquet(trg_val2k_fn)
  print(f"saved to {trg_val2k_fn}")

  trg_dpo85k_df = trg_df[82000:]
  print(f"length of trg_dpo85k_df: {len(trg_dpo85k_df)}")
  trg_dpo85k_df = pd.DataFrame(trg_dpo85k_df)
  trg_dpo85k_df.to_parquet(trg_dpo85k_fn)
  print(f"saved to {trg_dpo85k_fn}")


def shuf_resp_order():
  src_fn = "/mnt/mnhotzc/data/po/po_tulu_valid2k.parquet"
  trg_fn = "/mnt/mnhotzc/data/po/po_tulu_valid2k_shuf.parquet"
  src_df = pd.read_parquet(src_fn)
  trg_df = []
  n_ans1 = n_ans2 = 0
  for idx in range(len(src_df)):
    row_dict = src_df.iloc[idx].to_dict()
    if random.random() > 0.5:
      resp1 = row_dict["response2"]
      resp2 = row_dict["response1"]
      ans = "2"
      n_ans2 += 1
    else:
      resp1 = row_dict["response1"]
      resp2 = row_dict["response2"]
      ans = "1"
      n_ans1 += 1
    new_dict = {
      "data_source": "skywork_raw_v2",
      "question": row_dict["question"],
      "response1": resp1,
      "response2": resp2,
      "answer": ans,
      "extra_info": row_dict["extra_info"],
    }
    trg_df.append(new_dict)
  trg_df = pd.DataFrame(trg_df)
  print(f"length of trg_df: {len(trg_df)}")
  trg_df.to_parquet(trg_fn)
  print(f"saved to {trg_fn}")
  print(f"n_ans1: {n_ans1}, n_ans2: {n_ans2}")


if __name__ == '__main__':
  # main()
  shuf_resp_order()