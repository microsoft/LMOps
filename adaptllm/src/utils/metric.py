from sklearn.metrics import f1_score
import numpy as np
from src.utils import qa_utils
import nereval
import math

def compute_headline(data):
    # 1. collect weighted f1 per class
    merged_dic={}
    class_num = 9
    for class_id in range(class_num):
        merged_dic[class_id] = {}
        merged_dic[class_id]['True'] = []
        merged_dic[class_id]['Pred'] = []
    
    for entry in data:
        class_id = entry['class_id']
        merged_dic[class_id]['True'].append(entry['label'])
        merged_dic[class_id]['Pred'].append(entry['pred'])
    
    f1_list=[]
    for class_id in range(class_num):
        f1 = f1_score(y_true = merged_dic[class_id]['True'], 
                    y_pred = merged_dic[class_id]['Pred'],
                    average = 'weighted')
        f1_list.append(f1)
    
    # 2. average by class num
    return sum(f1_list)/class_num


def compute_ner(data):
    f1_list = []
    for entry in data:
        true = qa_utils.parse_ner(entry['label'][0])
        pred = qa_utils.parse_ner(entry['pred'])
        f1 = nereval.evaluate([true], [pred])
        f1_list.append(f1)
    return sum(f1_list)/len(f1_list)


def compute_ConvFinQA(labels, preds):
    true_count = 0
    for i in range(len(labels)):
        label_set = qa_utils.normalized_num(labels[i][0])
        pred_set = qa_utils.normalized_num(preds[i])
        if len(pred_set)>0 and len(label_set)>0:
            label_num = label_set[0]
            for pred_num in pred_set:
                if math.isclose(pred_num,label_num,abs_tol = 1e-2):
                    true_count += 1
                    break
    res = true_count/len(labels)
    return res


def simple_accuracy(labels, preds):
    return (preds == labels).mean()


def multi_label_acc(labels, preds):
    '''
        simplified acc for multi-label classification
        acc = 1 if pred in label(s) set
    '''
    def flatten_label(label):
        if len(label) == 1 and isinstance(label[0],list):
            label = label[0]
        return label
    labels = [flatten_label(l) for l in labels]
    res = [int(preds[i] in labels[i]) for i in range(len(preds))]
    return sum(res)/len(res) 

def compute_metrics(metric,labels,preds,data):
    assert len(preds) == len(labels)
    if metric == 'weighted_F1':
        weighted_F1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
        return {'weighted_F1': weighted_F1*100}
    elif metric == 'Headline':
        multi_class_weighted_F1 = compute_headline(data)
        return {'multi_class_weighted_F1':multi_class_weighted_F1*100}
    elif metric == 'NER':
        entity_f1 = compute_ner(data)
        return {'entity_level_f1':entity_f1*100}
    elif metric == 'ConvFinQA':
        ConvFinQA_em = compute_ConvFinQA(labels=labels, preds=preds)
        return {'ConvFinQA_em': ConvFinQA_em*100}
    elif metric == 'micro_F1':
        F1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        return {'micro_F1': F1*100}
    elif metric == 'acc':
        acc = simple_accuracy(labels=labels, preds=preds)
        return {'acc': acc*100}
    elif metric == "micro_F1_and_macro_F1":
        micro_F1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        macro_F1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        return {'micro_F1': micro_F1*100, 'macro_F1': macro_F1*100}
    elif metric == 'multi_label_acc':
        acc = multi_label_acc(labels=labels, preds=preds)
        return {'multi_label_acc': acc*100}

def compute_scores(metric,data):
    preds = [entry['pred'] for entry in data]
    labels = [entry['label'] for entry in data]
    if not isinstance(preds[0], str):
        preds = np.array(preds)
        labels = np.array(labels)
    scores = compute_metrics(metric,labels=labels, preds=preds, data=data)
    return scores