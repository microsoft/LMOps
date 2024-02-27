import torch
import os
from transformer_trainer import TransformerTrainer
from perceptron_trainer import PerceptronTrainer
import re

MAX_EPOCHS = 1000

class EvalPolicyTrainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        if args.data_names == "tinystory":
            base_trainer_cls = TransformerTrainer
        elif args.data_names == "linear":
            base_trainer_cls = PerceptronTrainer
        else:
            raise NotImplementedError(args.data_names)
        
        self.base_trainer = base_trainer_cls(args, device)
        
    def train(self, wandb_name=None):
        eval_gamma_epochs = self.args.eval_gamma_epochs
        policy_name = self.args.policy_name
        if re.match(eval_gamma_epochs, "c"):
            print(f"### Training Model with Constant Policy ###")
            self.base_trainer.train(wandb_name="constant", calc_CT=(not self.args.eval_no_CT))
            self.base_trainer.reload_model()
        
        print(eval_gamma_epochs)
        
        gamma_epochs_list = [i for i in range(MAX_EPOCHS) if re.match(eval_gamma_epochs, str(i))]
        for gamma_epoch in gamma_epochs_list:
            gamma = torch.load(os.path.join(self.args.load_gamma, f"epoch_{gamma_epoch}", "opt_gamma.pt"))
            gamma = gamma.to(self.device)
            print(f"### Training Model with Policy {policy_name} at {gamma_epoch} ###")
            self.base_trainer.train(gamma=gamma, wandb_name="opt_gamma_{}/{}".format(
                policy_name, gamma_epoch), calc_CT=(not self.args.eval_no_CT))
            self.base_trainer.reload_model()
