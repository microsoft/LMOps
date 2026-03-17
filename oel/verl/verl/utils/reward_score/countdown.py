import re
import ast
import operator
from collections import Counter
from typing import Optional, Union, List


def parse_output_text(output_text: str) -> Optional[str]:
    """
    Extract the content from the last \boxed{...} if present.
    If not present, return the original text as per user's prompt.
    """
    if output_text is None:
        return None
    matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
    if matches:
        return matches[-1].strip()
    return output_text.strip()


def validate_expression(x_list: List[int], expr: str, target: int,
                        min_val: int = 0, max_val: int = 999,
                        check_intermediate: bool = False) -> bool:
    """
    Validate that `expr`:
      • uses exactly the numbers in `x_list`
      • employs only + - * /  (integer division)
      • lands on `target`
      • keeps results inside [min_val, max_val] if `check_intermediate` is True
    """
    OPS = {ast.Add: operator.add,
           ast.Sub: operator.sub,
           ast.Mult: operator.mul,
           ast.Div: operator.floordiv,
           ast.FloorDiv: operator.floordiv}

    if not isinstance(expr, str):
        return False
    if '\x00' in expr:
        return False
    expr = expr.strip()
    if not expr:
        return False

    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, ValueError, TypeError):
        return False

    used = Counter()

    def _eval(node):
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = OPS.get(type(node.op))
            if not op:
                raise ValueError("Unsupported operator")
            
            # Integer division check: must be exact and not by zero
            if op is operator.floordiv:
                if right == 0 or left % right != 0:
                    raise ValueError("Invalid division")
            
            result = op(left, right)
            if check_intermediate and not (min_val <= result <= max_val):
                raise ValueError("Intermediate result out of bounds")
            return result
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            used[node.value] += 1
            return node.value
        else:
            raise ValueError("Unsupported node type")

    try:
        final = _eval(tree.body)
    except (ValueError, AttributeError, ZeroDivisionError, TypeError):
        return False

    return (final == target and
            used == Counter(x_list))


def verify(solution_str: str, numbers: List[int], target: int, **kwargs) -> tuple[bool, str]:
    """
    Verify if the solution is correct.
    """
    pred_expr = parse_output_text(solution_str)
    
    min_val = kwargs.get('min_val', 0)
    max_val = kwargs.get('max_val', 999)
    check_intermediate = kwargs.get('check_intermediate', False)
    
    correct = validate_expression(numbers, pred_expr, target, 
                                  min_val=min_val, 
                                  max_val=max_val, 
                                  check_intermediate=check_intermediate)
    return correct, pred_expr


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
        ground_truth: The ground truth data, expected to be [[numbers], [target]]
    """
    # Parse ground_truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = ast.literal_eval(ground_truth)
        except:
            return {"score": 0.0, "acc": False, "pred": None}
    
    try:
        numbers = ground_truth[0]
        # ground_truth is [[1,2,3], [5]], so ground_truth[1] is [5]
        target = ground_truth[1][0] if isinstance(ground_truth[1], list) else ground_truth[1]
    except (IndexError, TypeError):
        return {"score": 0.0, "acc": False, "pred": None}

    # Verify the solution
    correct, pred = verify(solution_str, numbers, target, **kwargs)
    
    # Following math_dapo.py format: return 1.0 or -1.0
    reward = 1.0 if correct else -1.0
    
    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
    }
