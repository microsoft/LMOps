import pandas as pd

# read data frames from parquet files
df = pd.read_parquet("deepscaler/hdfs_data/train.parquet")
output_fn = "/mnt/jiaxin/data/rlvr/math_nl.parquet"

# get the first 10 rows of the data frame
# print(df.head(10))

new_df = []
for i in range(len(df)):
  data = df.iloc[i].to_dict()
  if i == 0:
    print(data)
  data["reward_model"] = {
    "ground_truth": "",
    "style": "model",
  }
  if i == 0:
    print(data)
  new_df.append(data)

new_df = pd.DataFrame(new_df)
# save the new data frame to a parquet file
new_df.to_parquet(output_fn)
print(f"Saved to {output_fn}")

