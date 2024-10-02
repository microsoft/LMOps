import sys
base_path = sys.argv[1]
sys.path.append(base_path)
import os
import torch

paths = [
    "results/ol/bruce/mistral/ms20-1k/cc-16384-flan-squad-1024/bs32-lr0.008-G32-N16-NN2-tn16384-dn1024-e100-tNone/static-optsgd-scdconstant-olr1.0-oe10-wm0-ct10_ga/10-20-7/2",
    "results/ol/bruce/mistral/ms20-10k/cc-16384-flan-squad-1024/bs32-lr0.008-G32-N16-NN2-tn16384-dn1024-e100-tNone/static-optsgd-scdconstant-olr0.2-oe10-wm0-ct10_ga/10-20-7/2",
    "results/ol/bruce/mistral/ms20-100k/cc-16384-flan-squad-1024/bs32-lr0.008-G32-N16-NN2-tn16384-dn1024-e100-tNone/static-optsgd-scdconstant-olr0.2-oe10-wm0-ct10_ga/10-20-7/1"
]

all_indices = []
for path in paths:
    opt_gamma = torch.load(os.path.join(path, "opt_gamma.pt"), map_location="cpu")
    # indices where gamma is non zero
    indices = torch.nonzero(opt_gamma).squeeze().tolist()
    all_indices.append(indices)
    
print([len(indices) for indices in all_indices])

all_indices = [set(indices) for indices in all_indices]

# merge indices
merged_indices = all_indices[0]
print(len(merged_indices))
for indices in all_indices[1:]:
    merged_indices = merged_indices.union(indices)
    print(len(merged_indices))
    
print()
print(len(all_indices[0].intersection(all_indices[1])))
print(len(all_indices[1].intersection(all_indices[2])))
