from datasets import load_dataset

class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls

        return adder

class Base_Task(object): 
    '''
    base class for loading filled-in input/output from huggingface dataset
    '''
    def __init__(self):
        super().__init__()
        # sum of inference batch size of all the GPUs
        # increase it for faster speed, and decrease it if OOM
        self.inf_bsz = 1

    def get_question(self, entry):
        return entry["input"]
    
    def get_input_strs(self, entry, class_num):
        question = self.get_question(entry)
        return [question] * class_num

    def get_answers(self, entry):
        if 'options' in entry:
            return [' ' + o for o in entry['options']]
        else:
            return [' ' + entry['label']] if isinstance(entry['label'], str) else [' ' + l for l in entry['label']]

    def get_label(self, entry):
        if 'gold_index' in entry:
            return entry['gold_index']
        else:
            return [' ' + entry['label']] if isinstance(entry['label'], str) else [' ' + l for l in entry['label']]

    def get_answer(self, entry):
        if 'gold_index' in entry:
            return ' ' + entry['options'][entry['gold_index']]
        else:
            return ' ' + entry['label'] if isinstance(entry['label'], str) else ' ' + entry['label'][0]


task_map = App()

# ================================ Finance =================================
@task_map.add("FPB")
class FPB(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "weighted_F1"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'FPB', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("FiQA_SA")
class FiQA_SA(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "weighted_F1"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'FiQA_SA', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("Headline")
class Headline(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "Headline"
        self.inf_bsz = 32 # larger bsz for faster speed, reduce it if OOM

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'Headline', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("NER")
class NER(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "NER"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'NER', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("ConvFinQA")
class ConvFinQA(Base_Task):
    def __init__(self):
        super().__init__()
        self.metric = "ConvFinQA"
        self.class_num = 1
        self.inf_bsz = 16 # larger bsz for faster speed, reduce it if OOM
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'ConvFinQA', cache_dir=cache_dir)
        return dataset['test']

 
# ================================ Medicine =================================
@task_map.add("ChemProt")
class ChemProt(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 13
        self.metric = "micro_F1"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/medicine-tasks', 'ChemProt', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("RCT")
class RCT(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 5
        self.metric = "micro_F1"
        self.inf_bsz = 4 # larger bsz for faster speed, reduce it if OOM

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/medicine-tasks', 'RCT', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("MQP")
class MQP(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "acc"
        self.inf_bsz = 10
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/medicine-tasks', 'MQP', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("USMLE")
class USMLE(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 4
        self.metric = "acc"
        self.inf_bsz = 2
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/medicine-tasks', 'USMLE', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("PubMedQA")
class PubMedQA(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "acc"
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/medicine-tasks', 'PubMedQA', cache_dir=cache_dir)
        return dataset['test']


# ================================ Law =================================
@task_map.add('SCOTUS')
class SCOTUS(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 13
        self.metric = "micro_F1_and_macro_F1"
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/law-tasks', 'SCOTUS', cache_dir=cache_dir)
        return dataset['test']


@task_map.add('CaseHOLD')
class CaseHOLD(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = None
        self.metric = "micro_F1_and_macro_F1"
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/law-tasks', 'CaseHOLD', cache_dir=cache_dir)
        return dataset['test']
    
    def get_input_strs(self, entry, class_num):
        return entry['input_options'] 
    
    def get_answers(self, entry=None):
        answers = [entry['output']] * len(entry['input_options'])
        return answers 

@task_map.add('UNFAIR_ToS')
class UNFAIR_ToS(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 8+1 # +1 for none
        self.metric = "multi_label_acc" # simplified acc, if pred in labels, acc=True
        self.inf_bsz = 4 # larger bsz for faster speed, reduce it if OOM
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/law-tasks', 'UNFAIR_ToS', cache_dir=cache_dir)
        return dataset['test']