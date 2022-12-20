from .BaseTask import BaseTask
from .BoolQ import BoolQ
from .CB import CB
from .COPA import COPA
from .SST2 import SST2
from .RTE import RTE
from .AGNews import AGNews
from .SST5 import SST5
from .Subj import Subj
from .TREC import TREC
from .ARCC import ARCC
from .ARCE import ARCE
from .DBPedia import DBPedia
from .HellaSwag import HellaSwag
from .MR import MR
from .OBQA import OBQA
from .PIQA import PIQA
from .StoryCloze import StoryCloze
from .Winogrande import Winogrande
from .WiC import WiC
from .WSC import WSC
from .MultiRC import MultiRC


dataset_dict = {
    'cb': CB,
    'copa': COPA,
    'sst2': SST2,
    'rte': RTE,
    'agnews': AGNews,
    'sst5': SST5,
    'subj': Subj,
    'trec': TREC,
    'arcc': ARCC,
    'arce': ARCE,
    'dbpedia': DBPedia,
    'hellaswag': HellaSwag,
    'mr': MR,
    'obqa': OBQA,
    'piqa': PIQA,
    'storycloze': StoryCloze,
    'winogrande': Winogrande,
    'boolq': BoolQ,
    'wic': WiC,
    'wsc': WSC,
    "multirc": MultiRC
}

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](*args, **kwargs)


