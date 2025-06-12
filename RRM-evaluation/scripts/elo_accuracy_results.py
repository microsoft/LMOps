# 重新导入所需库并重新执行逻辑（因为状态已重置）
from evalica import elo, Winner
import json
from collections import defaultdict
from pathlib import Path
import ipdb
import random
random.seed(0)  

file_paths = [
    "/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_7b_RRM_judgement/7b_merged_results.json",
]

all_file_paths = [
    f"/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_candidates/test_qwen25-math-cot_-1_seed{i}_t0.6_method1_runno_prompt_sample_five_s0_e-1.jsonl"
    for i in range(8)
]

def load_all_match_details(match_files):
    details_by_question = defaultdict(list)
    for path in match_files:
        with open(path, "r") as f:
            match_data = json.load(f)
        for group in match_data.values():
            for d in group["details"]:
            
                qid = d["question_id"]
                details_by_question[qid].append(d)
    return details_by_question

def calculate_elo_by_question(details_by_question):
    qid_top_model = {}
    for qid, matches in details_by_question.items():
        random.shuffle(matches)
        if not matches:
            continue
        rivals_1 = []
        rivals_2 = []
        outcomes = []
        for m in matches:
            rivals_1.append(str(m["rival_1"]))
            rivals_2.append(str(m["rival_2"]))
            winner = m["winner"]
            if winner == m["rival_1"]:
                outcomes.append(Winner.X)
            elif winner == m["rival_2"]:
                outcomes.append(Winner.Y)
            else:
                outcomes.append(Winner.Draw)
        result = elo(rivals_1, rivals_2, outcomes)
        top_model = result.scores.idxmax()
        qid_top_model[qid] = int(top_model)
    return qid_top_model

def evaluate_accuracy_by_top_model(qid_top_model, all_file_paths):
    model_predictions = [{} for _ in range(8)]
    model_scores = [{} for _ in range(8)]
    ground_truth = {}

    for i, path in enumerate(all_file_paths):
        with open(path, "r") as f:
            for line in f:
                j = json.loads(line)
                idx = j["idx"]
                pred = j["pred"][0] if j["pred"] else None
                gt = j["gt"]
                score = j["score"][0] if j["score"] else None
                model_predictions[i][idx] = pred
                model_scores[i][idx] = score
                ground_truth[idx] = gt

    correct = 0
    total = 0
    for qid, model_idx in qid_top_model.items():
        pred = model_predictions[model_idx].get(qid, None)
        gt = ground_truth.get(qid, None)
        score = model_scores[model_idx].get(qid, None)
        if score is not None:
            total += 1
            if score == 1: 
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, total, correct


details_by_question = load_all_match_details(file_paths)
qid_top_model = calculate_elo_by_question(details_by_question)
accuracy, total, correct = evaluate_accuracy_by_top_model(qid_top_model, all_file_paths)
print(f"Accuracy: {accuracy:.4f}, Total: {total}, Correct: {correct}")

predictions_save_path = "/home/v-jiaxinguo/finetuning/scripts/MATH_Qwen_7b_RRM_judgement/7b_elo_predictions.json"
with open(predictions_save_path, "w") as f:
    json.dump(qid_top_model, f, indent=2)
