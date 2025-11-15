import pandas as pd
import random

df = pd.read_parquet("/mnt/jiaxin/data/rlvr/webinst_rlvr-sl_train.parquet")
output_fn_nl = "/mnt/jiaxin/data/rlvr/webinst_rlvr-nl_train.parquet"


# print(df.head(10))
# for i in range(10):
#   data = df.iloc[i].to_dict()
#   print(data)

new_df_nl = []
for i in range(len(df)):
    data = df.iloc[i].to_dict()
    if i == 0:
        print(data)
    # if data["reward_model"]["style"] == "model":
    #     continue
    data["reward_model"] = {
        "style": "model",
        "ground_truth": ""
    }
    
    new_df_nl.append(data)

print(f"original df: {len(df)}, new_df_nl: {len(new_df_nl)}")

new_df_nl = pd.DataFrame(new_df_nl)
# save the new data frame to a parquet file
new_df_nl.to_parquet(output_fn_nl)
print(f"Saved to {output_fn_nl}")
    