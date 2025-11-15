import pandas as pd
import random

df = pd.read_parquet("/mnt/jiaxin/data/rlvr/webinst_rlvr-sl_train.parquet")
output_fn_hl = "/mnt/jiaxin/data/rlvr/webinst_rlvr-hl_train.parquet"


# print(df.head(10))
# for i in range(10):
#   data = df.iloc[i].to_dict()
#   print(data)

new_df_hl = []
for i in range(len(df)):
    data = df.iloc[i].to_dict()
    if i == 0:
        print(data)
    if data["reward_model"]["style"] == "model":
        continue
    
    new_df_hl.append(data)

print(f"original df: {len(df)}, new_df_hl: {len(new_df_hl)}")

new_df_hl = pd.DataFrame(new_df_hl)
# save the new data frame to a parquet file
new_df_hl.to_parquet(output_fn_hl)
print(f"Saved to {output_fn_hl}")
    