import random
import torch

def load_train_dataset(dataset,size=None,listify=True):
    if size is not None and size<len(dataset['train']):
        data = dataset['train']
        rand = random.Random(x=42)
        index_list = list(range(len(data))) 
        rand.shuffle(index_list) #shuffle index_list 
        x = data.select(index_list[:size])

    else:
        x = dataset['train']
    if listify:
        return list(x)
    else:
        return x

def pad2sameLen(
    values,
    pad_idx=0,
    left_pad=False
):
    """Convert a list of 1d tensors into a padded 2d tensor.
    ensuring same lengths
    """
    size = max(v.shape[-1] for v in values)
    if left_pad:
        res=torch.stack([torch.nn.functional.pad(v,(size-v.shape[-1],0),value=pad_idx) for v in values])
    else:
        res=torch.stack([torch.nn.functional.pad(v,(0,size-v.shape[-1]),value=pad_idx) for v in values])
    return res
