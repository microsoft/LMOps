from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np
from rouge import Rouge
from src.utils import qa_utils
from datasets import load_metric
import re

class App:
    def __init__(self):
        self.functions = {}

    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func

        return adder

    def __getitem__(self, __name: str):
        return self.functions[__name]


metric_dict = App()


@metric_dict.add("rouge")
def rouge(preds, labels, return_list=False):
    # https://github.com/pltrdy/rouge
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        if "\n" not in preds[i]:
            preds[i] += "\n"  # to ensure rouge metrics
        if "\n" not in labels[i]:
            labels[i] += "\n"
        scores = r.get_scores(preds[i], labels[i])[0]
        r1s.append(scores["rouge-1"]["f"])
        r2s.append(scores["rouge-2"]["f"])
        rls.append(scores["rouge-l"]["f"])
    if return_list:  # used for scoring data
        return r1s
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    return r1, r2, rl


@metric_dict.add("squad")
def squad(labels, preds, return_list=False):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    if return_list:  # used for scoring data
        em, f1 = qa_utils.qa_metrics(labels, preds, return_list=True)
        return f1
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


@metric_dict.add("trivia_qa")
def trivia_qa(labels, preds, return_list=False):
    """Computes TriviaQA metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and preds
    """
    labels = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_trivia_qa(p) for p in preds]
    if return_list:  # used for scoring data
        em, f1 = qa_utils.qa_metrics(labels, preds, return_list=True)
        return f1
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


@metric_dict.add("simple_accuracy")
def simple_accuracy(preds, labels, return_list=False):
    if isinstance(preds[0], str):
        labels = [label.strip() for label in labels]
        preds = [pred.strip() for pred in preds]
    res = [int(preds[i] == labels[i]) for i in range(len(preds))]
    if return_list:
        return res
    acc = sum(res) / len(res)
    return acc

@metric_dict.add("pubmed_qa_acc")
def pubmed_qa_acc(preds, labels, return_list=False):
    pattern=r'([.\s]*)(the answer is)(.*)' 
    regex=re.compile(pattern,re.IGNORECASE)

    res_list = []
    for i, pred in enumerate(preds):
        label = labels[i]
        if len(regex.findall(pred))>0:
            answer = regex.findall(pred)[-1][-1].lower()
            if "yes" in answer:
                acc = 1 if label=='yes' else 0
            elif "no" in answer:
                acc = 1 if label=='no' else 0
            elif "maybe" in answer:
                acc = 1 if label=='maybe' else 0
            else:
                acc = 0
        else:
            answer = None
            acc = 0
        res_list.append(acc)
    if return_list:
        return res_list
    return sum(res_list)/len(res_list)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return acc, f1, (acc + f1) / 2


def acc_and_matthews_corrcoef(preds, labels):
    acc = simple_accuracy(preds, labels)
    mcc = matthews_corrcoef(y_true=labels, y_pred=preds)
    return acc, mcc


def compute_bleu(preds, labels):
    BLEU = load_metric("bleu")
    predictions = [[ch for ch in text] for text in preds]
    references = [[[ch for ch in label]] for label in labels]
    return BLEU.compute(predictions=predictions, references=references)


def compute_metrics(metric, labels, preds):
    assert len(preds) == len(labels)
    if metric == "simple_accuracy":
        return {"acc": simple_accuracy(preds, labels) * 100}
    elif metric == "rouge":
        r1, r2, rl = rouge(preds, labels)
        return {"r1": r1 * 100, "r2": r2 * 100, "rl": rl * 100}
    elif metric == "acc_and_f1":
        acc, f1, acc_f1 = acc_and_f1(preds, labels)
        return {"acc": acc * 100, "f1": f1 * 100, "acc_and_f1": acc_f1 * 100}
    elif metric == "acc_and_matthews_corrcoef":
        acc, mcc = acc_and_matthews_corrcoef(preds, labels)
        return {"acc": acc * 100, "mcc": mcc * 100}
    elif metric == "f1":
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {"f1": f1 * 100}
    elif metric == "squad":
        em, f1 = squad(labels=labels, preds=preds)
        return {"em": em, "f1": f1}
    elif metric == "trivia_qa":
        em, f1 = trivia_qa(labels=labels, preds=preds)
        return {"em": em, "f1": f1}
    elif metric == "bleu":
        bleu = compute_bleu(preds=preds, labels=labels)
        return {"bleu": bleu["bleu"] * 100}
    elif metric == "pubmed_qa_acc":
        acc = pubmed_qa_acc(preds=preds, labels=labels)
        return {"pubmed_qa_acc": acc * 100}


def compute_scores(metric, data):
    preds = [entry["pred"] for entry in data]
    labels = [entry["label"] for entry in data]
    if not isinstance(preds[0], str):
        preds = np.array(preds)
        labels = np.array(labels)
    scores = compute_metrics(metric, labels=labels, preds=preds)
    return scores