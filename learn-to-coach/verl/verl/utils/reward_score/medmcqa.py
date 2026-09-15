from typing import Optional, Union, List

def compute_score(
    solution_str: str,
    ground_truth: Union[str, List],
    method: str = 'strict', # for future flexibility
    **kwargs
) -> dict:
    """
    Compute the reward score for a countdown solution.
    
    Args:
        solution_str: The solution string from the model
        ground_truth: The ground truth data
    """
    CUT = 16
    pred = solution_str[:CUT]
    correct = (ground_truth in pred) or (ground_truth.lower() in pred)
    
    # Following math_dapo.py format: return 1.0 or -1.0
    reward = 1.0 if correct else -1.0
    
    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
    }
