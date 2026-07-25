"""
Stage 1: Generate model responses for fuzzy benchmarks using vLLM.

Usage:
    python scripts/eval/gen_fuzzy.py \
        --benchmark alpacaeval2 \
        --model /tmp/Qwen3-1.7B \
        --output_dir /tmp/el/results/exp/global_step_25/eval_fuzzy/

Benchmarks: alpacaeval2, wildbench, arena_hard_v2, creativewritingv3
"""

import argparse
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


DATA_BASE = os.environ.get("EL_DATA_ROOT", "/tmp/el/data")

BENCHMARKS = ["alpacaeval2", "wildbench", "arena_hard_v2", "creativewritingv3"]


def load_benchmark(benchmark):
    """Load benchmark prompts and metadata.

    Returns list of {"inst_id", "messages", "metadata"} where messages is
    a chat-format list of dicts.
    """
    if benchmark == "alpacaeval2":
        path = os.path.join(DATA_BASE, "alpacaeval2", "alpaca_eval_gpt4_baseline.json")
        with open(path) as f:
            data = json.load(f)
        return [
            {
                "inst_id": f"alpacaeval2_{i}",
                "messages": [{"role": "user", "content": d["instruction"]}],
                "metadata": {"baseline": d["output"], "instruction": d["instruction"]},
            }
            for i, d in enumerate(data)
        ]

    elif benchmark == "wildbench":
        path = os.path.join(DATA_BASE, "wildbench", "v2.json")
        with open(path) as f:
            data = json.load(f)
        return [
            {
                "inst_id": f"wildbench_{i}",
                "messages": d["conversation_input"],
                "metadata": {
                    "references": d.get("references", {}),
                    "checklist": d.get("checklist", []),
                    "conversation_input": d["conversation_input"],
                },
            }
            for i, d in enumerate(data)
        ]

    elif benchmark == "arena_hard_v2":
        path = os.path.join(DATA_BASE, "arena_hard_v2", "prompts.json")
        with open(path) as f:
            data = json.load(f)
        return [
            {
                "inst_id": d.get("uid", f"arena_hard_v2_{i}"),
                "messages": [{"role": "user", "content": d["prompt"]}],
                "metadata": {
                    "reference": d.get("reference", ""),
                    "reference_model": d.get("reference_model", "unknown"),
                    "category": d.get("category", "hard_prompt"),
                    "prompt": d["prompt"],
                },
            }
            for i, d in enumerate(data)
        ]

    elif benchmark == "creativewritingv3":
        path = os.path.join(DATA_BASE, "creativewritingv3", "creative_writing_prompts_v3.json")
        with open(path) as f:
            prompts = json.load(f)
        items = []
        for key, obj in prompts.items():
            base = obj.get("writing_prompt", "")
            seeds = obj.get("seed_modifiers", [])
            n_iter = min(3, len(seeds))
            for i in range(n_iter):
                final = base.replace("<SEED>", seeds[i])
                items.append(
                    {
                        "inst_id": f"creativewritingv3_{key}_{i + 1}",
                        "messages": [{"role": "user", "content": final}],
                        "metadata": {"base_prompt": base, "seed_modifier": seeds[i]},
                    }
                )
        return items

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for fuzzy benchmarks")
    parser.add_argument("--benchmark", type=str, required=True, choices=BENCHMARKS)
    parser.add_argument("--model", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # Per-benchmark max_tokens (following GOLF convention)
    if args.benchmark in ("arena_hard_v2",):
        max_tokens = 8192
    else:
        max_tokens = args.max_tokens  # default 4096
    max_model_len = 4096 + max_tokens  # 4096 reserved for prompt

    output_file = os.path.join(args.output_dir, f"{args.benchmark}_samples.json")
    if os.path.exists(output_file):
        print(f"Output file already exists, will overwrite: {output_file}")

    print(f"Loading benchmark: {args.benchmark}")
    instances = load_benchmark(args.benchmark)
    print(f"Loaded {len(instances)} instances")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Tokenize prompts
    print("Applying chat template and tokenizing...")
    token_ids_list = []
    for inst in instances:
        token_ids = tokenizer.apply_chat_template(
            inst["messages"],
            add_generation_prompt=True,
            tokenize=True,
        )
        token_ids_list.append(token_ids)

    print(f"max_tokens={max_tokens}, max_model_len={max_model_len}")

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
    )

    # Generate
    print(f"Generating responses (temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens})...")
    from vllm import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list]
    outputs = llm.generate(prompts, sampling_params)

    # Build results
    results = []
    for inst, output in zip(instances, outputs):
        response_text = output.outputs[0].text
        # Extract the last user message as the "prompt" field for readability
        last_user_msg = ""
        for msg in reversed(inst["messages"]):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        results.append(
            {
                "inst_id": inst["inst_id"],
                "prompt": last_user_msg,
                "response": response_text,
                "metadata": inst["metadata"],
            }
        )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} samples to {output_file}")


if __name__ == "__main__":
    main()
