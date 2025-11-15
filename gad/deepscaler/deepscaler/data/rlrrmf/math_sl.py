import pandas as pd
import random

# read data frames from parquet files
df = pd.read_parquet("deepscaler/hdfs_data/train.parquet")
output_fn_sl = "/mnt/jiaxin/data/rlvr/math_sl.parquet"
output_fn_hl = "/mnt/jiaxin/data/rlvr/math_hl.parquet"

# get the first 10 rows of the data frame
# print(df.head(10))

new_df_sl = []
new_df_hl = []
for i in range(len(df)):
  data = df.iloc[i].to_dict()
  if i == 0:
    print(data)
  if random.random() < 0.5:
    data["reward_model"] = {
      "ground_truth": "",
      "style": "model",
    }
  else:
    new_df_hl.append(data)
  if i == 0:
    print(data)
  new_df_sl.append(data)

print(f"new_df_sl: {len(new_df_sl)}")
print(f"new_df_hl: {len(new_df_hl)}")

# new_df = pd.DataFrame(new_df)
# # save the new data frame to a parquet file
# new_df.to_parquet(output_fn)
# print(f"Saved to {output_fn}")

new_df_sl = pd.DataFrame(new_df_sl)
new_df_hl = pd.DataFrame(new_df_hl)
# save the new data frame to a parquet file
new_df_sl.to_parquet(output_fn_sl)
print(f"Saved to {output_fn_sl}")
new_df_hl.to_parquet(output_fn_hl)
print(f"Saved to {output_fn_hl}")

