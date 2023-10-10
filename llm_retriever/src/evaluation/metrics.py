import torch

from typing import List
from rouge import Rouge
from sklearn.metrics import f1_score

from evaluation import qa_utils
from logger_config import logger


@torch.no_grad()
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def batch_mrr(output: torch.tensor, target: torch.tensor) -> float:
    assert len(output.shape) == 2
    assert len(target.shape) == 1
    sorted_score, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    assert rank.shape[0] == output.shape[0]

    rank = rank + 1
    mrr = torch.sum(100 / rank.float()) / rank.shape[0]
    return mrr.item()


## =========================================================================== ##

# Copy from https://github.com/microsoft/LMOps/tree/main/uprise/src/utils


def rouge(preds, labels):
    # https://github.com/pltrdy/rouge
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        if '\n' not in preds[i]: preds[i] += '\n'
        if '\n' not in labels[i]: labels[i] += '\n'  # avoid empty string
        scores = r.get_scores(preds[i], labels[i])[0]
        r1s.append(scores["rouge-1"]['f'])
        r2s.append(scores["rouge-2"]['f'])
        rls.append(scores["rouge-l"]['f'])
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    return r1, r2, rl


def squad(labels, preds):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


def trivia_qa(labels, preds):
    """Computes TriviaQA metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and preds
    """
    labels = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_trivia_qa(p) for p in preds]
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


def simple_accuracy(preds, labels):
    if isinstance(preds[0], str):
        labels = [label.lower().strip() for label in labels]
        preds = [pred.lower().strip() for pred in preds]
    res = [int(preds[i] == labels[i]) for i in range(len(preds))]
    acc = sum(res) / len(res)
    return acc


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    # Currently only MRPC & QQP use this metric
    f1 = f1_score(y_true=labels, y_pred=preds, pos_label='Yes')
    return acc, f1, (acc + f1) / 2


def compute_metrics(metric, labels, preds):
    assert len(preds) == len(labels)
    if metric == 'simple_accuracy':
        return {'acc': simple_accuracy(preds, labels) * 100}
    elif metric == 'rouge':
        r1, r2, rl = rouge(preds, labels)
        return {'r1': r1 * 100, 'r2': r2 * 100, 'rl': rl * 100}
    elif metric == 'acc_and_f1':
        acc, f1, acc_f1 = acc_and_f1(preds, labels)
        return {'acc': acc * 100, 'f1': f1 * 100, 'acc_and_f1': acc_f1 * 100}
    elif metric == 'f1':
        f1 = f1_score(y_true=labels, y_pred=preds, pos_label='Yes')
        return {'f1': f1 * 100}
    elif metric == 'squad':
        em, f1 = squad(labels=labels, preds=preds)
        return {'em': em, 'f1': f1}
    elif metric == 'trivia_qa':
        em, f1 = trivia_qa(labels=labels, preds=preds)
        return {'em': em, 'f1': f1}
    else:
        raise ValueError(metric)
