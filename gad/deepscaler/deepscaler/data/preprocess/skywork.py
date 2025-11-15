import json
import random
from datasets import load_dataset, DatasetDict

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

def process_data(dataset):
    processed_data = []
    for example in dataset:
        # Extract the user content from "chosen"
        user_content = example["chosen"][0]["content"]  # Use the first user turn as the problem
        chosen_assistant = " ".join([turn["content"] for turn in example["chosen"] if turn["role"] == "assistant"])
        rejected_assistant = " ".join([turn["content"] for turn in example["rejected"] if turn["role"] == "assistant"])

        # Randomly decide the order of chosen and rejected solutions
        if random.random() < 0.5:
            problem = f"{user_content}\n\nChoose the better answer from the following two responses:\nSolution 1: {chosen_assistant}\nSolution 2: {rejected_assistant}"
            answer = "1"  # Chosen is first
        else:
            problem = f"{user_content}\n\nChoose the better answer from the following two responses:\nSolution 1: {rejected_assistant}\nSolution 2: {chosen_assistant}"
            answer = "2"  # Rejected is first

        # Organize into the target format
        processed_data.append({"problem": problem, "answer": answer})

    return processed_data

# Process train and test datasets
train_processed = process_data(new_ds["train"])
test_processed = process_data(new_ds["test"])

# Save as JSON files
with open("/home/v-jiaxinguo/finetuning/deepscaler/deepscaler/data/train/skywork.json", "w", encoding="utf-8") as f:
    json.dump(train_processed, f, ensure_ascii=False, indent=4)

with open("/home/v-jiaxinguo/finetuning/deepscaler/deepscaler/data/test/skywork.json", "w", encoding="utf-8") as f:
    json.dump(test_processed, f, ensure_ascii=False, indent=4)

print("Train and test sets have been saved as train.json and test.json")
