import os
import re
import json
import random
import time
import argparse
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import openai

# ============================================================
# OpenAI client
# ============================================================

class OpenAIClient:
    def __init__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running this evaluator.")

        client_kwargs = {"max_retries": 0, "timeout": 600}
        if os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**client_kwargs)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def call(self, prompt, max_tokens=2048, temperature=0.0, json_mode=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return self._call_messages(messages, max_tokens, temperature, json_mode)

    def call_with_system(self, system_prompt, user_prompt, max_tokens=2048, temperature=0.0, json_mode=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_messages(messages, max_tokens, temperature, json_mode)

    def _call_messages(self, messages, max_tokens=2048, temperature=0.0, json_mode=False):
        max_retry = 50
        cur_retry = 0
        response_format = {"type": "json_object"} if json_mode else None
        while cur_retry <= max_retry:
            try:
                t0 = time.time()
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                if response_format:
                    kwargs["response_format"] = response_format
                completion = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - t0
                if elapsed > 600:
                    raise TimeoutError(f"API call took {elapsed:.0f}s (>600s)")
                results = completion.choices[0].message.content
                return results or ""
            except openai.RateLimitError:
                time.sleep(1)
            except Exception as e:
                err_str = str(e)
                if "ResponsibleAIPolicyViolation" in err_str:
                    print(f"Retry {cur_retry}: ResponsibleAIPolicyViolation, skipping")
                    return ""
                if "max_tokens" in err_str and "max_completion_tokens" in err_str:
                    try:
                        kwargs_alt = dict(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=max_tokens,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None,
                        )
                        if response_format:
                            kwargs_alt["response_format"] = response_format
                        completion = self.client.chat.completions.create(**kwargs_alt)
                        results = completion.choices[0].message.content
                        return results
                    except openai.RateLimitError:
                        time.sleep(1)
                    except Exception as e2:
                        err_str2 = str(e2)
                        if "ResponsibleAIPolicyViolation" in err_str2:
                            return ""
                        print(f"Retry {cur_retry}: {err_str2[:200]}")
                        cur_retry += 1
                else:
                    print(f"Retry {cur_retry}: {err_str[:200]}")
                    cur_retry += 1
        return ""


# ============================================================
# Utility: shorten text
# ============================================================

def shorten(text, max_words=None):
    if max_words is None:
        max_words = SHORTEN_MAX_WORDS
    if max_words <= 0 or not text:
        return text or ""
    words = text.split(" ")
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "... (truncated)"
    return text


# ============================================================
# AlpacaEval2
# ============================================================

ALPACAEVAL2_TEMPLATE = """I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": \"\"\"{instruction}\"\"\",
}

Here are the outputs of the models:
[
    {
        "model": "A",
        "answer": \"\"\"{model_output}\"\"\"
    },
    {
        "model": "B",
        "answer": \"\"\"{reference}\"\"\"
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
{
    "output": [
        {"model": <model-name>, "rank": <model-rank>},
        {"model": <model-name>, "rank": <model-rank>}
    ]
}
Your response must be a valid json dictionary and contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""


def _parse_alpacaeval_result(result_str):
    """Parse AlpacaEval pairwise JSON ranking. Returns winning model letter or None."""
    if not result_str:
        return None
    try:
        for entry in json.loads(result_str)["output"]:
            model = entry["model"]
            rank = int(entry["rank"])
            if rank == 1 and model in ("A", "B"):
                return model
    except Exception:
        pass
    return None


def judge_alpacaeval2(oai_client, sample, idx):
    instruction = shorten(sample["metadata"]["instruction"])
    candidate = shorten(sample["response"])
    baseline = shorten(sample["metadata"]["baseline"])

    # Randomize position
    if random.random() < 0.5:
        a_text, b_text = candidate, baseline
        candidate_pos = "A"
    else:
        a_text, b_text = baseline, candidate
        candidate_pos = "B"

    prompt = ALPACAEVAL2_TEMPLATE.replace("{instruction}", instruction).replace("{model_output}", a_text).replace("{reference}", b_text)
    output = oai_client.call(prompt, max_tokens=128, temperature=0.0, json_mode=True)
    winner = _parse_alpacaeval_result(output)
    failed = not output or winner is None

    preferred = 1 if winner == candidate_pos else 0
    delta_len = len(candidate.split()) - len(baseline.split())
    return idx, preferred, delta_len, output, failed


# ============================================================
# _WildBench
# ============================================================

WILDBENCH_PAIRWISE_TEMPLATE = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of the responses generated by two AI models.
We will provide you with the user query and a pair of AI-generated responses (Response A and Response B).
You should first read the user query and the conversation history carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{history}

<|end_of_history|>

## Current User Query
<|begin_of_query|>

{user_query}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{candidate_A}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{candidate_B}

<|end_of_response_B|>

# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your assessment to the checklist.

## Rules

You should compare the above two responses based on your analysis of the user queries and the conversation history.
You should first write down your analysis and the checklist that you used for the evaluation, and then provide your assessment according to the checklist.
There are five choices to give your final assessment: ["A++", "A+", "A=B", "B+", "B++"], which correspond to the following meanings:

- `A++`: Response A is much better than Response B.
- `A+`: Response A is only slightly better than Response B.
- `A=B`: Response A and B are of the same quality. Please use this choice sparingly.
- `B+`: Response B is only slightly better than Response A.
- `B++`: Response B is much better than Response A.


## Output Format
First, please output your analysis for each model response, and then summarize your assessment to three aspects: "reason A=B", "reason A>B", and "reason B>A", and finally make your choice for the final assessment.

Please provide your evaluation results in the following json format by filling in the placeholders in []:
```
{
    "analysis of A": "[analysis of Response A]",
    "analysis of B": "[analysis of Response B]",
    "reason of A=B": "[where Response A and B perform equally well]",
    "reason of A>B": "[where Response A is better than Response B]",
    "reason of B>A": "[where Response B is better than Response A]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```"""


WILDBENCH_REWARD_MAP = {
    "A++": 100, "A+": 50, "A=B": 0, "B+": -50, "B++": -100,
}


def _parse_wildbench_choice(result_str):
    """Parse WildBench JSON choice. Returns choice string or None."""
    if not result_str:
        return None
    try:
        parsed = json.loads(result_str)
        return parsed.get("choice", None)
    except Exception:
        # Fallback regex
        m = re.search(r'"choice"\s*:\s*"([^"]*)"', result_str)
        if m:
            return m.group(1)
    return None


def judge_wildbench(oai_client, sample, idx):
    metadata = sample["metadata"]

    # Reconstruct history from conversation_input
    conv = metadata.get("conversation_input", [])
    if conv:
        history_list, last_query_msg = conv[:-1], conv[-1]
        history = ""
        for entry in history_list:
            if entry["role"] == "user":
                history += "USER: " + entry["content"] + "\n\n"
            else:
                history += "ASSISTANT: " + entry["content"] + "\n\n"
        user_query = last_query_msg.get("content", sample["prompt"])
    else:
        history = ""
        user_query = sample["prompt"]

    user_query = shorten(user_query)

    reference = metadata.get("references", {}).get("gpt-4", "")
    checklist = metadata.get("checklist", [])
    checklist_md = "\n".join(f"- {c}" for c in checklist)
    candidate = shorten(sample["response"])
    reference = shorten(reference)

    # Randomize AB
    if random.random() < 0.5:
        a_text, b_text = candidate, reference
        candidate_pos = "A"
    else:
        a_text, b_text = reference, candidate
        candidate_pos = "B"

    prompt = (WILDBENCH_PAIRWISE_TEMPLATE
              .replace("{history}", history)
              .replace("{user_query}", user_query)
              .replace("{candidate_A}", a_text)
              .replace("{candidate_B}", b_text)
              .replace("{checklist}", checklist_md))

    output = oai_client.call(prompt, max_tokens=1024, temperature=0.0, json_mode=True)
    choice = _parse_wildbench_choice(output)
    failed = not output or choice is None or choice.strip() not in WILDBENCH_REWARD_MAP

    multiplier = 1 if candidate_pos == "A" else -1
    reward = -100  # default if unparseable (consistent with RLMT)
    if choice and choice.strip() in WILDBENCH_REWARD_MAP:
        reward = WILDBENCH_REWARD_MAP[choice.strip()] * multiplier

    return idx, reward, output, failed


# ============================================================
# Arena Hard V2
# ============================================================

ARENA_HARD_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
    "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. "
    "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. "
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

ARENA_HARD_CREATIVE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
    "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. "
    "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. "
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

ARENA_HARD_USER_TEMPLATE = (
    "<|User Prompt|>\n{question}\n\n"
    "<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
)

_ARENA_VERDICT_PATTERNS = [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"]


def _parse_arena_verdict(judgment_text):
    if not judgment_text:
        return None
    upper = judgment_text.upper()
    for pattern in _ARENA_VERDICT_PATTERNS:
        matches = re.findall(pattern, upper)
        matches = [m for m in matches if m]
        if matches:
            return matches[-1].strip("\n")
    return None


def _arena_score_round(verdict, flipped):
    if verdict is None:
        return 0.5
    verdict = verdict.replace(" ", "")
    if not flipped:
        # A=baseline, B=candidate
        if verdict in ("B>A", "B>>A"):
            return 1.0
        if verdict == "A=B":
            return 0.5
        if verdict in ("A>B", "A>>B"):
            return 0.0
    else:
        # A=candidate, B=baseline
        if verdict in ("A>B", "A>>B"):
            return 1.0
        if verdict == "A=B":
            return 0.5
        if verdict in ("B>A", "B>>A"):
            return 0.0
    return 0.5


def judge_arena_hard_v2(oai_client, sample, idx):
    metadata = sample["metadata"]
    question = metadata["prompt"]
    baseline = metadata.get("reference", "")
    category = metadata.get("category", "hard_prompt")
    candidate = sample["response"]

    system_prompt = ARENA_HARD_CREATIVE_SYSTEM_PROMPT if category == "creative_writing" else ARENA_HARD_SYSTEM_PROMPT

    # Round 1: A=baseline, B=candidate
    user1 = ARENA_HARD_USER_TEMPLATE.format(question=question, answer_a=baseline, answer_b=candidate)
    output1 = oai_client.call_with_system(system_prompt, user1, max_tokens=3072, temperature=0.0)
    v1 = _parse_arena_verdict(output1)
    s1 = _arena_score_round(v1, flipped=False)

    # Round 2: A=candidate, B=baseline
    user2 = ARENA_HARD_USER_TEMPLATE.format(question=question, answer_a=candidate, answer_b=baseline)
    output2 = oai_client.call_with_system(system_prompt, user2, max_tokens=3072, temperature=0.0)
    v2 = _parse_arena_verdict(output2)
    s2 = _arena_score_round(v2, flipped=True)

    win_rate = (s1 + s2) / 2.0
    failed = v1 is None or v2 is None
    return idx, win_rate, {"round1": output1, "round2": output2}, failed


# ============================================================
# CreativeWritingV3
# ============================================================

CW3_JUDGE_PROMPT = """You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{test_model_response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- In the output, write the metric names exactly as below so they can be parsed.

- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.

- For these criteria, lower is better:
{lower_is_better_criteria}

- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

{creative_writing_criteria}"""

CW3_CRITERIA = [
    "Adherence to Instructions", "Believable Character Actions", "Nuanced Characters",
    "Consistent Voice/Tone of Writing", "Imagery and Descriptive Quality", "Elegant Prose",
    "Emotionally Engaging", "Emotionally Complex", "Coherent", "Meandering",
    "Weak Dialogue", "Tell-Don't-Show", "Unsurprising or Uncreative", "Amateurish",
    "Purple Prose", "Overwrought", "Incongruent Ending Positivity",
    "Unearned Transformations", "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally", "Overall Reader Engagement", "Overall Impression",
]

CW3_NEGATIVE_CRITERIA = [
    "Unearned Transformations", "Incongruent Ending Positivity", "Overwrought",
    "Purple Prose", "Amateurish", "Unsurprising or Uncreative",
    "Tell-Don't-Show", "Weak Dialogue", "Meandering",
]


def _parse_cw3_scores(judge_response):
    if not judge_response:
        return {}
    scores = {}
    p1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    p2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    for matches in [re.findall(p1, judge_response), re.findall(p2, judge_response)]:
        for name, val in matches:
            name = name.strip()
            val = float(val)
            if val <= 20:
                scores[name] = val
    return scores


def _compute_cw3_score(scores):
    vals = []
    for metric, val in scores.items():
        if isinstance(val, (int, float)):
            if metric in CW3_NEGATIVE_CRITERIA:
                val = 20.0 - val
            if val <= 20:
                vals.append(val)
    if not vals:
        return None
    return sum(vals) / len(vals) * 5.0  # scale to 0-100


def _truncate_words(text, max_words=3200):
    if not text:
        return ""
    it = re.finditer(r'\S+', text)
    m = next(itertools.islice(it, max_words - 1, None), None)
    return text[:m.end()] if m else text


def judge_creativewritingv3(oai_client, sample, idx):
    base_prompt = sample["metadata"].get("base_prompt", sample["prompt"])
    candidate = _truncate_words(sample["response"], 3200)

    criteria_str = "\n".join("- " + c for c in CW3_CRITERIA)
    negative_str = ", ".join(CW3_NEGATIVE_CRITERIA)

    prompt = CW3_JUDGE_PROMPT.format(
        writing_prompt=base_prompt,
        test_model_response=candidate,
        creative_writing_criteria=criteria_str,
        lower_is_better_criteria=negative_str,
    )

    output = oai_client.call(prompt, max_tokens=2048, temperature=0.0)
    scores = _parse_cw3_scores(output)
    final_score = _compute_cw3_score(scores)
    failed = final_score is None
    if final_score is None:
        final_score = 0.0

    return idx, final_score, output, failed


# ============================================================
# Main: process checkpoints
# ============================================================

def process_checkpoint(oai_client, eval_dir, output_dir, benchmark, max_workers=32):
    samples_path = os.path.join(eval_dir, f"{benchmark}_samples.json")
    if not os.path.exists(samples_path):
        print(f"  Skipping: {samples_path} not found")
        return

    with open(samples_path) as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} samples from {samples_path}")

    scores_filename = f"scores_4o_{benchmark}{SCORE_SUFFIX}.json"
    scores_path = os.path.join(output_dir, scores_filename)

    os.makedirs(output_dir, exist_ok=True)

    if benchmark == "alpacaeval2":
        _process_alpacaeval2(oai_client, samples, output_dir, max_workers)
    elif benchmark == "wildbench":
        _process_wildbench(oai_client, samples, output_dir, max_workers)
    elif benchmark == "arena_hard_v2":
        _process_arena_hard_v2(oai_client, samples, output_dir, max_workers)
    elif benchmark == "creativewritingv3":
        _process_creativewritingv3(oai_client, samples, output_dir, max_workers)


def _process_alpacaeval2(oai_client, samples, output_dir, max_workers):
    results = {}
    delta_lens = {}
    failed_indices = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_alpacaeval2, oai_client, s, i): i for i, s in enumerate(samples)}
        done = 0
        for future in as_completed(futures):
            idx, preferred, delta_len, output, failed = future.result()
            results[idx] = preferred
            delta_lens[idx] = delta_len
            if failed:
                failed_indices.append(idx)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t_start
                print(f"    [{done}/{len(samples)}] speed={done/elapsed:.2f} samples/s")

    # Compute win_rate
    prefs = [results.get(i, 0) for i in range(len(samples))]
    win_rate = sum(prefs) / len(prefs) * 100.0

    # Compute lc_win_rate via logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        rows = []
        for i in range(len(samples)):
            p = results.get(i, 0)
            dl = delta_lens.get(i, 0)
            rows.append({"model": "model", "delta_len": dl, "win": p})
            rows.append({"model": "reference", "delta_len": -dl, "win": 1 - p})
        df = pd.DataFrame(rows)
        X = pd.get_dummies(df["model"], prefix="model")
        X["delta_len"] = df["delta_len"]
        y = df["win"]
        clf = LogisticRegression(fit_intercept=True, solver="lbfgs")
        clf.fit(X, y)
        X_zero = X.copy()
        X_zero["delta_len"] = 0
        df["lc_pred"] = clf.predict_proba(X_zero)[:, 1]
        lc_win_rate = df[df["model"] == "model"]["lc_pred"].mean() * 100.0
    except Exception as e:
        print(f"  lc_win_rate computation failed: {e}, using raw win_rate")
        lc_win_rate = win_rate

    output = {
        "benchmark": "alpacaeval2",
        "win_rate": win_rate,
        "lc_win_rate": lc_win_rate,
        "total": len(samples),
        "failed_indices": sorted(failed_indices),
        "preferences": prefs,
    }
    with open(os.path.join(output_dir, f"scores_4o_alpacaeval2{SCORE_SUFFIX}.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  AlpacaEval2: win_rate={win_rate:.2f}, lc_win_rate={lc_win_rate:.2f}, failed={len(failed_indices)}")


def _process_wildbench(oai_client, samples, output_dir, max_workers):
    results = {}
    failed_indices = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_wildbench, oai_client, s, i): i for i, s in enumerate(samples)}
        done = 0
        for future in as_completed(futures):
            idx, reward, output, failed = future.result()
            results[idx] = reward
            if failed:
                failed_indices.append(idx)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t_start
                print(f"    [{done}/{len(samples)}] speed={done/elapsed:.2f} samples/s")

    rewards = [results.get(i, -100) for i in range(len(samples))]
    avg_reward = sum(rewards) / len(rewards)

    output = {
        "benchmark": "wildbench",
        "avg_reward": avg_reward,
        "total": len(samples),
        "failed_indices": sorted(failed_indices),
        "rewards": rewards,
    }
    with open(os.path.join(output_dir, f"scores_4o_wildbench{SCORE_SUFFIX}.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  WildBench: avg_reward={avg_reward:.2f}, failed={len(failed_indices)}")


def _process_arena_hard_v2(oai_client, samples, output_dir, max_workers):
    results = {}
    failed_indices = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_arena_hard_v2, oai_client, s, i): i for i, s in enumerate(samples)}
        done = 0
        for future in as_completed(futures):
            idx, win_rate, aux, failed = future.result()
            results[idx] = win_rate
            if failed:
                failed_indices.append(idx)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t_start
                print(f"    [{done}/{len(samples)}] speed={done/elapsed:.2f} samples/s")

    win_rates = [results.get(i, 0.5) for i in range(len(samples))]
    avg_win_rate = sum(win_rates) / len(win_rates)

    output = {
        "benchmark": "arena_hard_v2",
        "avg_win_rate": avg_win_rate,
        "total": len(samples),
        "failed_indices": sorted(failed_indices),
        "win_rates": win_rates,
    }
    with open(os.path.join(output_dir, f"scores_4o_arena_hard_v2{SCORE_SUFFIX}.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Arena Hard V2: avg_win_rate={avg_win_rate:.4f}, failed={len(failed_indices)}")


def _process_creativewritingv3(oai_client, samples, output_dir, max_workers):
    results = {}
    failed_indices = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(judge_creativewritingv3, oai_client, s, i): i for i, s in enumerate(samples)}
        done = 0
        for future in as_completed(futures):
            idx, score, output, failed = future.result()
            results[idx] = score
            if failed:
                failed_indices.append(idx)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t_start
                print(f"    [{done}/{len(samples)}] speed={done/elapsed:.2f} samples/s")

    scores = [results.get(i, 0.0) for i in range(len(samples))]
    avg_score = sum(scores) / len(scores)

    output = {
        "benchmark": "creativewritingv3",
        "avg_score": avg_score,
        "total": len(samples),
        "failed_indices": sorted(failed_indices),
        "scores": scores,
    }
    with open(os.path.join(output_dir, f"scores_4o_creativewritingv3{SCORE_SUFFIX}.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"  CreativeWritingV3: avg_score={avg_score:.2f}, failed={len(failed_indices)}")


if __name__ == "__main__":
    VALID_BENCHMARKS = ["alpacaeval2", "wildbench", "arena_hard_v2", "creativewritingv3"]

    parser = argparse.ArgumentParser(description="Score fuzzy benchmark samples using GPT-4o")
    parser.add_argument("--benchmark", type=str, required=True, help="Comma-separated benchmarks, e.g. alpacaeval2,wildbench,arena_hard_v2,creativewritingv3")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--start_ckpt", type=int, required=True, help="Start checkpoint")
    parser.add_argument("--end_ckpt", type=int, required=True, help="End checkpoint")
    parser.add_argument("--step", type=int, default=25, help="Checkpoint step")
    parser.add_argument("--max_workers", type=int, default=64, help="Number of concurrent API calls")
    shorten_group = parser.add_mutually_exclusive_group()
    shorten_group.add_argument(
        "--no_shorten",
        dest="no_shorten",
        action="store_true",
        help="Disable truncation of long texts (default)",
    )
    shorten_group.add_argument(
        "--shorten",
        dest="no_shorten",
        action="store_false",
        help="Truncate long texts to 2000 words",
    )
    parser.set_defaults(no_shorten=True)
    parser.add_argument("--reval", type=int, default=0, help="Re-evaluation index; appends _revalN to score filenames")
    parser.add_argument("--base_dir", type=str, default=os.environ.get("EL_RESULT_ROOT", "/tmp/el/results"), help="Base directory")
    args = parser.parse_args()

    global SHORTEN_MAX_WORDS, SCORE_SUFFIX
    SCORE_SUFFIX = ""
    if args.no_shorten:
        SHORTEN_MAX_WORDS = 0
        SCORE_SUFFIX += "_full"
    else:
        SHORTEN_MAX_WORDS = 2000
    if args.reval > 0:
        SCORE_SUFFIX += f"_reval{args.reval}"

    benchmarks = [b.strip() for b in args.benchmark.split(",")]
    for b in benchmarks:
        assert b in VALID_BENCHMARKS, f"Unknown benchmark: {b}. Valid: {VALID_BENCHMARKS}"

    print("Initializing OpenAI client...")
    oai_client = OpenAIClient()
    print("Client ready.")

    ckpts = list(range(args.start_ckpt, args.end_ckpt + 1, args.step))
    print(f"Benchmarks: {benchmarks}, Experiment: {args.exp_name}, checkpoints: {ckpts}")

    for ckpt in ckpts:
        eval_dir = os.path.join(args.base_dir, args.exp_name, f"global_step_{ckpt}", "eval_fuzzy")
        output_dir = eval_dir  # Save scores alongside the samples
        for benchmark in benchmarks:
            print(f"\nProcessing {args.exp_name} step {ckpt} benchmark {benchmark}:")
            process_checkpoint(oai_client, eval_dir, output_dir, benchmark, max_workers=args.max_workers)

    print("\nAll done!")
