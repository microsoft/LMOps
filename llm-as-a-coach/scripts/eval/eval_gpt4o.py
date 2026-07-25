import os
import re
import json
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import openai


RUBRIC_BASED_PROMPT_TEMPLATE = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.

<prompt>
{instruction}
</prompt>

<response>
{response}
</response>

<rubrics>
{rubric_list_string}
</rubrics>

First, analyze the response against each rubric item, discussing how well the response meets or fails each criterion. Then, provide your final score as an integer between 1 and 10, wrapped in <score> and </score> tags.
Example ending:
<score> your_integer_score_from_1_to_10 </score>

Your evaluation:"""


class OpenAIClient:
    def __init__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running this evaluator.")

        client_kwargs = {"max_retries": 0, "timeout": 600}
        if os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**client_kwargs)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def call(self, prompt, max_tokens=2048, temperature=0.7):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        max_retry = 50
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                t0 = time.time()
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                elapsed = time.time() - t0
                if elapsed > 600:
                    raise TimeoutError(f"API call took {elapsed:.0f}s (>600s)")
                results = completion.choices[0].message.content
                usage = completion.usage
                return results, usage
            except openai.RateLimitError:
                time.sleep(1)
            except Exception as e:
                if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                    try:
                        t0 = time.time()
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=max_tokens,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None,
                        )
                        elapsed = time.time() - t0
                        if elapsed > 600:
                            raise TimeoutError(f"API call took {elapsed:.0f}s (>600s)")
                        results = completion.choices[0].message.content
                        usage = completion.usage
                        return results, usage
                    except openai.RateLimitError:
                        time.sleep(1)
                    except Exception as e2:
                        err_str = str(e2)
                        if "ResponsibleAIPolicyViolation" in err_str:
                            print(f"Retry {cur_retry}: ResponsibleAIPolicyViolation, skipping")
                            return "", None
                        print(f"Retry {cur_retry}: {err_str[:200]}")
                        cur_retry += 1
                else:
                    err_str = str(e)
                    if "ResponsibleAIPolicyViolation" in err_str:
                        print(f"Retry {cur_retry}: ResponsibleAIPolicyViolation, skipping")
                        return "", None
                    print(f"Retry {cur_retry}: {err_str[:200]}")
                    cur_retry += 1
        return "", None


def score_sample(oai_client, sample, idx):
    """Score a single sample using GPT-4o with rubric-based prompt."""
    prompt = RUBRIC_BASED_PROMPT_TEMPLATE.format(
        instruction=sample["prompt"],
        response=sample["response"],
        rubric_list_string=sample.get("rubric_list_string", ""),
    )
    output, usage = oai_client.call(prompt, max_tokens=2048, temperature=0.7)
    if not output:
        return idx, 1.0, output, usage
    score = 1.0
    score_match = re.search(r'<score>(.*?)</score>', output, re.DOTALL | re.IGNORECASE)
    if score_match:
        score_number_match = re.search(r'(\d+(?:\.\d+)?)', score_match.group(1).strip())
        if score_number_match:
            try:
                score = float(score_number_match.group(1))
                score = max(1.0, min(10.0, score))
            except ValueError:
                score = 1.0
    return idx, score, output, usage


def process_checkpoint(oai_client, eval_dir, output_dir, max_workers=32, resume=False, reval=0, src_suffix=""):
    """Score all samples in a checkpoint's random_samples.json."""
    scores_suffix = f"_reval{reval}" if reval > 0 else ""
    scores_suffix = f"{src_suffix}{scores_suffix}"
    samples_path = os.path.join(eval_dir, "random_samples.json")
    if not os.path.exists(samples_path):
        print(f"  Skipping: {samples_path} not found")
        return

    with open(samples_path, encoding="utf-8") as f:
        content = f.read()
    try:
        samples = json.loads(content)
    except json.JSONDecodeError:
        try:
            samples = json.JSONDecoder().raw_decode(content)[0]
        except json.JSONDecodeError as e:
            print(f"  ERROR: Failed to parse JSON: {e}")
            return

    print(f"  Loaded {len(samples)} samples from {samples_path}")

    # Resume support
    existing_scores = {}
    scores_path = os.path.join(output_dir, f"scores_4o{scores_suffix}.json")
    if resume and os.path.exists(scores_path):
        with open(scores_path) as f:
            existing = json.load(f)
        existing_scores = {i: s for i, s in enumerate(existing.get("scores", []))}
        print(f"  Resuming: {len(existing_scores)} scores already done")

    # Prepare tasks
    tasks = []
    for idx, sample in enumerate(samples):
        if idx in existing_scores:
            continue
        tasks.append((idx, sample))

    print(f"  Need to score {len(tasks)} samples ({len(existing_scores)} already done)")

    results = dict(existing_scores)
    if tasks:
        t_start = time.time()
        all_prompt_tokens = []
        all_completion_tokens = []
        all_total_tokens = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(score_sample, oai_client, sample, idx): idx
                for idx, sample in tasks
            }
            done_count = 0
            for future in as_completed(futures):
                idx, score, output, usage = future.result()
                results[idx] = score
                done_count += 1
                if usage:
                    all_prompt_tokens.append(getattr(usage, 'prompt_tokens', 0) or 0)
                    all_completion_tokens.append(getattr(usage, 'completion_tokens', 0) or 0)
                    all_total_tokens.append(getattr(usage, 'total_tokens', 0) or 0)
                if done_count % 50 == 0:
                    elapsed = time.time() - t_start
                    speed = done_count / elapsed
                    print(f"    [{done_count}/{len(tasks)}] speed={speed:.2f} samples/s, score={score}")
                    if all_prompt_tokens:
                        print(f"    --- Token stats (n={len(all_prompt_tokens)}) ---")
                        print(f"    prompt_tokens:     avg={sum(all_prompt_tokens)/len(all_prompt_tokens):.0f}, max={max(all_prompt_tokens)}")
                        print(f"    completion_tokens: avg={sum(all_completion_tokens)/len(all_completion_tokens):.0f}, max={max(all_completion_tokens)}")
                        print(f"    total_tokens:      avg={sum(all_total_tokens)/len(all_total_tokens):.0f}, max={max(all_total_tokens)}")
                if done_count % 500 == 0:
                    _save_scores(results, len(samples), output_dir, scores_suffix)

    _save_scores(results, len(samples), output_dir, scores_suffix)

    all_scores = [results.get(i, 1.0) for i in range(len(samples))]
    valid_count = sum(1 for s in all_scores if s != 1.0)
    avg_score = sum(all_scores) / len(all_scores)
    print(f"  Done: avg_score={avg_score:.4f}, valid={valid_count}/{len(samples)}")


def _save_scores(results, total, output_dir, scores_suffix=""):
    """Save scores to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    all_scores = [results.get(i, 1.0) for i in range(total)]
    avg_score = sum(all_scores) / len(all_scores)
    valid_count = sum(1 for s in all_scores if s != 1.0)
    output = {
        "scores": all_scores,
        "avg_score": avg_score,
        "avg_score_normalized": avg_score / 10.0,
        "valid_ratio": valid_count / total if total > 0 else 0.0,
        "total": total,
    }
    with open(os.path.join(output_dir, f"scores_4o{scores_suffix}.json"), "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score evaluation samples using GPT-4o")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--start_ckpt", type=int, required=True, help="Start checkpoint")
    parser.add_argument("--end_ckpt", type=int, required=True, help="End checkpoint")
    parser.add_argument("--step", type=int, default=25, help="Checkpoint step")
    parser.add_argument("--max_workers", type=int, default=64, help="Number of concurrent API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing scores")
    parser.add_argument("--reval", type=int, default=0, help="Re-evaluation index; 0=default scores_4o.json, N=scores_4o_revalN.json")
    parser.add_argument("--eval_suffix", type=str, default="evaluation", help="Source eval dir suffix, e.g. evaluation_trainset")
    parser.add_argument("--base_dir", type=str, default=os.environ.get("EL_RESULT_ROOT", "/tmp/el/results"), help="Base directory")
    args = parser.parse_args()

    print("Initializing OpenAI client...")
    oai_client = OpenAIClient()
    print("Client ready.")

    ckpts = list(range(args.start_ckpt, args.end_ckpt + 1, args.step))
    print(f"Experiment: {args.exp_name}, checkpoints: {ckpts}")

    # Determine source json suffix for trainset etc.
    eval_src_suffix = ""
    if args.eval_suffix != "evaluation":
        eval_src_suffix = args.eval_suffix.replace("evaluation", "").strip("_")
        if eval_src_suffix:
            eval_src_suffix = f"_{eval_src_suffix}"

    for ckpt in ckpts:
        eval_dir = os.path.join(args.base_dir, args.exp_name, f"global_step_{ckpt}", args.eval_suffix)
        output_dir = os.path.join(args.base_dir, args.exp_name, f"global_step_{ckpt}", "evaluation_4o")
        print(f"\nProcessing {args.exp_name} step {ckpt} (source: {args.eval_suffix}):")
        process_checkpoint(oai_client, eval_dir, output_dir, max_workers=args.max_workers, resume=args.resume, reval=args.reval, src_suffix=eval_src_suffix)

    print("\nAll done!")
