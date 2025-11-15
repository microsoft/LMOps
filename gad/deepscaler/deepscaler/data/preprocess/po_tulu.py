import datasets
import pandas as pd
from tqdm import tqdm

def main():

  ds = datasets.load_dataset("allenai/llama-3.1-tulu-3-405b-preference-mixture")["train"]

  # rm80k_fn = "/mnt/mnhotzc/data/po/po4o_rmtk_reward80k_train.parquet"
  trg_df = []
  # trg_fn = "/mnt/mnhotzc/data/po/po_tulu_reward80k_train.parquet"
  rm80k_fn = "/mnt/mnhotzc/data/po/po4o_rmtk_valid2k.parquet"
  trg_fn = "/mnt/mnhotzc/data/po/po_tulu_valid2k.parquet"

  ds_rm80k = pd.read_parquet(rm80k_fn)
  query_set = set()
  for idx in range(len(ds_rm80k)):
    row_dict = ds_rm80k.iloc[idx].to_dict()
    query_set.add(row_dict["question"].strip().replace(" ", ""))
  
  resp_len_sum = count = 0

  print(f"query_set len: {len(query_set)}")
  for indx, example in tqdm(enumerate(ds)):
    promt = example["prompt"].strip().replace(" ", "")
    if promt not in query_set:
      continue

    new_dict = {
      "data_source": "skywork_raw_v2",
      "question": example["prompt"],
      "response1": " ".join([turn["content"] for turn in example["chosen"] if turn["role"] == "assistant"]),
      "response2": " ".join([turn["content"] for turn in example["rejected"] if turn["role"] == "assistant"]),
      "answer": "1",
      "extra_info": {'index': indx, 'split': 'train'}
    }
    resp_len_sum += len(new_dict["response1"]) + len(new_dict["response2"])
    count += 2
    if idx ==0:
      print(new_dict)

    trg_df.append(new_dict)
  
  trg_df = pd.DataFrame(trg_df)
  print(f"length of trg_df: {len(trg_df)}")
  trg_df.to_parquet(trg_fn)
  print(f"saved to {trg_fn}")
  print(f"response_len_avg: {resp_len_sum/count}")
    


if __name__ == "__main__":
  main()