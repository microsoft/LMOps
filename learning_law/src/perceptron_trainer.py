import torch
import os

from perceptron_model import PerceptronModel
from base_trainer import BaseTrainer


class PerceptronTrainer(BaseTrainer):
    def __init__(self, args, device) -> None:
        super(PerceptronTrainer, self).__init__(args, device)

    def get_tokenizer(self):
        return None
    
    def get_model(self):
        config = {
            "hidden_size": self.args.hidden_size
        }
        model = PerceptronModel(config).to(self.device)
        return model

    def get_data(self):
        train_data = torch.load(os.path.join(self.args.data_dir, "train.pt"))
        dev_data = torch.load(os.path.join(self.args.data_dir, "dev.pt"))
        test_data = torch.load(os.path.join(self.args.data_dir, "test.pt"))

        return train_data, dev_data, test_data

    def reform_data(self, data):
        return data[0].to(self.device), data[1].to(self.device)