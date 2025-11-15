import json
import random
from datasets import load_dataset, DatasetDict, Dataset

# Load dataset
ds = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

# Get dataset size
total_size = len(ds["train"])
train_size = 75000
test_size = total_size - train_size

# Split dataset
train_test_split = ds["train"].train_test_split(train_size=train_size, test_size=test_size, shuffle=True, seed=42)

# Combine into new dataset
new_ds = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

# Save the new dataset to disk
new_ds.save_to_disk("/mnt/blob/skywork-data/skywork_raw")
