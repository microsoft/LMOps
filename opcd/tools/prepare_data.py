from datasets import load_dataset
import os

dataset_names = [
    "ytz20/dapo_train",
    "ytz20/dapo_validation",
    "ytz20/dapo_test",

    "ytz20/sys_medmcqa_train",
    "ytz20/sys_medmcqa_test",
    "ytz20/sys_safety_train",
    "ytz20/sys_safety_test",
]

output_dir = "/tmp"

for name in dataset_names:
    try:
        print(f"Processing {name}...")

        short_name = name.split("/")[-1]
        output_path = os.path.join(output_dir, f"{short_name}.parquet")

        ds = load_dataset(name)
        data = ds["train"]
        data.to_parquet(output_path)

        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"Failed on {name}: {e}")