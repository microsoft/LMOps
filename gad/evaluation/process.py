import json
# from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from tqdm import tqdm
input_data_path = "/Users/bytedance/Desktop/ppo-long-cot/math-eval/new_data/math_level1to5_data.json"
# from parser import *
from utils import PROMPT_TEMPLATES

prompt_temp = PROMPT_TEMPLATES["qwen25-math-cot"]
input_template, output_template, splitter = (prompt_temp[0], prompt_temp[1], prompt_temp[2])

with open(input_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data[0])
for example in tqdm(data):
    # example["question"] = example['problem']
    # if example["question"] == "":
    #     continue
    # gt_cot, gt_ans = parse_ground_truth(example, "math") #！！！解析 gt，便于后续比较，关键
    # example["gt_ans"] = gt_ans
    example["input"] = input_template.format(input=example["question"]).strip()

    # if idx == args.start:
    #     print(full_prompt)

    # sample = {
    #     "idx": idx,
    #     "question": example["question"],
    #     "gt_cot": gt_cot,
    #     "gt": gt_ans,
output_path = input_data_path.replace(".json", "_processed_with_qwen_prompt.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)