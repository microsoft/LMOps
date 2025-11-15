import os
import pandas as pd


def load_filter_results():
  ij2res = {}
  filter_res_dir = "/mnt/msranlp_zc2/zechi/data/skyw_filter"
  for fn in os.listdir(filter_res_dir):
    if fn.startswith("filter_results_") and fn.endswith(".txt"):
      with open(os.path.join(filter_res_dir, fn), "r") as f:
        lines = f.readlines()
      print(f"Load {len(lines)} lines from {fn}")
      for line in lines:
        index, jndex, res = line.strip().split("\t")
        ij2res[(int(index), int(jndex))] = int(res)
  
  i2res = {}
  for (i, j), res in ij2res.items():
    if i not in i2res:
      i2res[i] = res
    else:
      i2res[i] += res

  # statistics
  print("filter statistics:")
  # n3 = n4 = 0
  nn = [0] * 5
  for i, res in i2res.items():
    nn[res] += 1
  # print(f"Total: {len(i2res)} 75% correct: {n3*100.0 / len(i2res)}, 100% correct: {n4*100.0 / len(i2res)} 50% correct: {len(i2res) - n3}")
  print(f"Total: {len(i2res)}, nn: {nn}")
  tot_percent = 0
  for i in range(5):
    print(f"i:{i}, percentage: {nn[i] * 100.0 / len(i2res)}")
    tot_percent += nn[i] * 100.0 / len(i2res)
  print(f"Total percentage: {tot_percent}")
  # for j in range(75000):
  #   if j not in i2res:
  #     print(f"Missing {j}")
  return i2res
        


def main():
  i2res = load_filter_results()
  exit(0)
  parquet_file = "/mnt/msranlp_zc2/zechi/data/skyw_filter/skywork_raw_v2/train.parquet"
  dataframe = pd.read_parquet(parquet_file)

  filtered_df = []
  missing_idx_n = 0
  for idx in range(len(dataframe)):
    if idx not in i2res:
      missing_idx_n += 1
      filtered_df.append(dataframe.iloc[idx])
    else:
      if i2res[idx] < 4:
        filtered_df.append(dataframe.iloc[idx])
  
  print(f"original dataframe length: {len(dataframe)}")
  print(f"length of filtered dataframe: {len(filtered_df)}")
  print(f"missing idx num: {missing_idx_n}")
  print(f"missing idx percentage: {missing_idx_n * 100.0 / len(dataframe)}")

  # save the filtered dataframe to a new parquet file
  filtered_df = pd.DataFrame(filtered_df)
  # filtered_df.to_parquet("/mnt/msranlp_zc2/zechi/data/skyw_filter/skywork_raw_v2/train_filtered_v1.parquet")


if __name__ == '__main__':
  main()