import pandas as pd

# Load the parquet file
df = pd.read_parquet("/mnt/jiaxin/data/rlvr/webinst_rlvr-sl_train.parquet")

# Display the first 20 rows of the dataframe
# print(df.head())
# Display the first 20 rows of the dataframe
for i in range(20):
    print(df.iloc[i])
    print()
