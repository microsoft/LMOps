import multiprocessing
import argparse
import numpy as np


global_state = {}
from dataclasses import dataclass


# from dataflow.core.lispress import try_round_trip
   
from src.utils import top_utils


def eval_single_mtop(pred: str,gold: str):
    return pred.strip()==gold
    pred_lf = top_utils.deserialize_top(pred)
    gold_lf = top_utils.deserialize_top(gold)
    if pred_lf is None:
        return pred==gold
    else:
        return pred_lf.serialize()==gold_lf.serialize()



def eval_single_smcalflow(pred: str,gold: str):
    return pred.strip()==gold
    pred_lispress = try_round_trip(pred)
    gold_lispress = try_round_trip(gold)
    return pred_lispress==gold_lispress

@dataclass
class GlobalState:
    converter = None
    matcher = None
    scorer = None

    def __post_init__(self):
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()





def eval_single(question, generated,decomposition,index):
    try:
        # print(f"Starting: {index}")
        if "#13" in generated:
            return False
        def try_invoke(func, graph, default=None):
            try:
                return func(graph)
            except Exception as ex:
                return default

        gold = format_qdmr(decomposition)
        pred = format_qdmr(generated.replace("  "," ").lower())
    
        decomp_lf = global_state.converter.convert(question_id=str(index), question_text=question, decomposition=pred.to_break_standard_string())
        gold_lf = global_state.converter.convert(question_id=str(index), question_text=question, decomposition=gold.to_break_standard_string())
        s = global_state.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf, graph2=gold_lf)
        return s
    except Exception as ex:
        # print(f"Failed on: {index} | 0")
        return False


def eval_many(questions, preds,golds,n_proc=None):
    def set_global_object():
        global global_state
        global_state = GlobalState()

    pool = multiprocessing.Pool(processes=n_proc,initializer=set_global_object)
    mrange = list(range(len(preds)))
    results = pool.starmap_async(eval_single, list(zip(questions, preds,golds,mrange)))
    results = results.get(None)

    return results


def eval_many_mtop(preds,golds,n_proc=None):

    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.starmap_async(eval_single_mtop, list(zip(preds,golds)))
    results = results.get(None)

    return results


def eval_many_smcalflow(preds,golds,n_proc=None):
    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.starmap_async(eval_single_smcalflow, list(zip(preds,golds)))
    results = results.get(None)

    return results
