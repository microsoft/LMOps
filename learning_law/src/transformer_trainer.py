import os
import torch.distributed as dist

from transformer_model import Transformer, ToyTokenizer
from base_trainer import BaseTrainer


class TransformerTrainer(BaseTrainer):
    def __init__(self, args, device) -> None:
        super(TransformerTrainer, self).__init__(args, device)
    
    def get_tokenizer(self):
        tokenizer = ToyTokenizer(os.path.join(self.args.data_dir, "vocab.pt"))
        return tokenizer
        
    def get_model(self):
        config = {
            "vocab_size": self.args.vocab_size,
            "max_len": self.args.max_length,
            "hidden_size": self.args.hidden_size,
            "num_head": self.args.num_head,
            "num_layers": self.args.num_layers,
        }
        
        model = Transformer(self.args, config).to(self.device)
        for p in model.parameters():
            dist.broadcast(p, 0)
        return model
    
    def reform_data(self, data):
        assert data.size(1) == self.max_length + 1
        input_ids = data[:, :-1].clone().to(self.device)
        labels = data[:, 1:].clone().to(self.device)
        # labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
