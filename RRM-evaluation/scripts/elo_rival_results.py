import json
import os
from collections import defaultdict
import re
import ipdb

file_paths = [os.path.join("/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_7b_RRM_judgement", name) for name in os.listdir("/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_7b_RRM_judgement") if name.endswith("with_shuffle_s0_e-1.jsonl") and "group" not in name]
file_paths_new = []
for filename in file_paths:
    #print(filename)
    match = re.search(r"seed(\d+).*?rival_(\d+)_(\d+)", filename)

    if match:
        seed = int(match.group(1))
        rival_1 = int(match.group(2))
        rival_2 = int(match.group(3))
    #print(f"Processing file: {filename} with seed: {seed}, rival_1: {rival_1}, rival_2: {rival_2}")
    #ipdb.set_trace()
    #print(10 * rival_1 + rival_2)
    if seed == int(10 * rival_2 + rival_1):
        #if rival_2 == rival_1 + 1:# or rival_2 == rival_1 + 2 or (rival_2 == 6 and rival_1 == 0):
        #print(1)
        #remove the file from the list
        file_paths_new.append(filename)
        #print(file_paths)
#file_paths = [os.path.join("/mnt/blob/output/actor/huggingface/math", name) for name in os.listdir("/mnt/blob/output/actor/huggingface/math") if name.endswith("with_shuffle_s0_e-1.jsonl") and "group" not in name]
#file_paths = [os.path.join("/mnt/blob/output/actor/huggingface/elo-shuffle-bug-result", name) for name in os.listdir("/mnt/blob/output/actor/huggingface/elo-shuffle-bug-result") if name.endswith("with_shuffle_s0_e-1.jsonl") and "group" not in name]
#file_paths = [os.path.join("/mnt/blob/output/deepverify-autosync_az/7b-d511-step720/math", name) for name in file_paths if name.endswith("with_shuffle_s0_e-1.jsonl") and "group" not in name]
file_paths = file_paths_new
file_paths.sort()

results = defaultdict(list)

for path in file_paths:
    filename = os.path.basename(path)
    parts = filename.split("_")
    rival_1 = int(parts[parts.index("rival") + 1])
    rival_2 = int(parts[parts.index("rival") + 2])
    print(f"Processing file: {filename} with rival_1: {rival_1}, rival_2: {rival_2}")

    with open(path, "r") as f:
        for line in f:
            rival_1 = int(parts[parts.index("rival") + 1])
            rival_2 = int(parts[parts.index("rival") + 2])
            data = json.loads(line)
            question_id = data.get("idx")
            is_shuffle = data.get("is_shuffle")
            if is_shuffle:
                rival_1, rival_2 = rival_2, rival_1  
            pred = data.get("pred", [""])[0]
            score = data.get("score", [None])[0]


            if pred == "Assistant1":
                winner = rival_1
            elif pred == "Assistant2":
                winner = rival_2
            else:
                winner = rival_1 if hash(line) % 2 == 0 else rival_2  
            original_rival_1 = int(parts[parts.index("rival") + 1])

            results[str(original_rival_1)].append({
                "question_id": question_id,
                "rival_1": rival_1,
                "rival_2": rival_2,
                "winner": winner,
                "pred": pred,
                "score": score
            })

output_path = "/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_7b_RRM_judgement/7b_merged_results.json"
with open(output_path, "w") as f:
    json.dump({rival_1: {"details": detail_list} for rival_1, detail_list in results.items()}, f, indent=2)
