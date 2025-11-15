import numpy as np
import torch
from verl import DataProto
from math_verify import parse, verify


def math_verify_reward_function(solution_str, ground_truth):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # We always take the final solution
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0
