from typing import List, Dict, Union

from inference import qa_utils
from logger_config import logger


def compute_em_and_f1(labels: List[List[str]], preds: List[str]) -> Dict[str, float]:
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    em, f1, em_scores, f1_scores = qa_utils.qa_metrics(labels, preds)  # em,f1

    return {'em': round(em, 3), 'f1': round(f1, 3)}


def compute_metrics_dict(labels: Union[List[str], List[List[str]]], preds: List[str], eval_metrics: str) -> Dict[str, float]:
    metric_names: List[str] = eval_metrics.split(',')
    metric_dict: Dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in ['em_and_f1', 'dpr']:
            metric_dict.update(compute_em_and_f1(labels, preds))
        elif metric_name == 'kilt':
            logger.warning('KILT metric requires run separate script')
        else:
            raise ValueError(f'Invalid metric: {metric_name}')

    return metric_dict
