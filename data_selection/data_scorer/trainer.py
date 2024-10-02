import os
import sys
import json
import torch
import random
import shutil
from time import time
from tqdm import tqdm
from numerize.numerize import numerize

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, accuracy_score

from utils import print_rank, all_gather
from data_utils import DataScorerDataset
from train_eval_utils.base_trainer import BaseTrainer

from .modeling import DataScorerModel


class DataScorerTrainer(BaseTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)
        self.min_offset = 0
        self.min_idx = 0
        self.set_datasets(do_train=do_train)
        if do_train:
            self.prepare_learning()
        self.setup_model_and_optimizer(set_optim=do_train)
        self.best_ckpt = None
        self.best_spr = 0
    
    def get_model(self, args=None, device=None):
        if args.do_train:
            model = DataScorerModel(
                args, device, args.model_path, bias=args.data_scorer_bias, encoding=args.data_scorer_encoding, head_type=args.data_scorer_head_type)        
        elif args.do_infer or args.do_eval:
            with open(os.path.join(args.model_path, "config.json")) as f:
                config = json.load(f)
            bias = config.get("bias", False) or args.data_scorer_bias
            encoding = config.get("encoding", None) or args.data_scorer_encoding
            model = DataScorerModel(
                args, "cpu", os.path.join(args.base_path, config["base_model_path"].strip("/")), bias=bias, encoding=encoding)
            model.load_state_dict(torch.load(os.path.join(args.model_path, "data_scorer_model.pt"), map_location="cpu"))
            model = model.to(self.device)
            if args.do_infer and args.torch_compile is not None:
                model.inference = torch.compile(model.inference, mode=args.torch_compile)
                        
        return model
    
    def set_datasets(self, args=None, do_train=True):
        args = args or self.args
        if do_train:
            self.train_dataset = DataScorerDataset(args, self.tokenizer, "train", args.data_dir, args.train_num, do_probe=True)
            self.print_and_save(f"train num: {len(self.train_dataset)}")
            self.eval_dataset = DataScorerDataset(args, self.tokenizer, "valid", args.data_dir, args.dev_num, do_probe=True)
        else:
            assert self.args.do_infer
            if os.path.exists(os.path.join(args.save, "state.json")):
                with open(os.path.join(args.save, "state.json")) as f:
                    state = json.load(f)
                self.min_offset = state["offset"]
                self.min_idx = state["idx"]
                self.print_and_save(f"Found existing state {state}")
            else:
                self.min_offset = 0
                self.min_idx = 0
            self.eval_dataset = DataScorerDataset(args, self.tokenizer, "data", args.data_dir, args.infer_num-self.min_offset, min_offset=self.min_offset)

    def first_print(self, model_batch, no_model_batch, save_name=""):
        if self.dp_rank == 0:
            print(model_batch["input_ids"][0])
            print(self.tokenizer.decode(model_batch["input_ids"][0].cpu().tolist()))
            print("attention_mask", model_batch["attention_mask"][0])
            print("labels", model_batch["labels"][0])
            print("poses", model_batch["pos"])
            print("idxs", no_model_batch["idx"])

    def compute_loss(self, model_batch, no_model_batch):
        return self.model(**model_batch), {}
        
    def evaluate_loss(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate)
        
        self.model.eval()
        all_losses = []
                    
        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(eval_dataloader, f"Evaluation", disable=(not self.dp_rank == 0)):
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                loss, loss_stat = self.compute_loss(model_batch, no_model_batch)
                all_losses.append(loss.clone())
        
        all_losses = torch.stack(all_losses, dim=0)
        all_losses = all_gather(all_losses, dim=1, op="stack").view(-1)
        avg_loss = all_losses.mean().item()
                
        if self.dp_rank == 0:
            res = {"avg_loss": avg_loss}
        else:
            res = None
        
        dist.barrier()
        return res
    
    def evaluate(self):
        res = self.evaluate_loss()
        infer_res = self.inference(os.path.join(self.args.save, f"eval_{self.global_steps}"))
        if self.dp_rank == 0:
            res.update(infer_res)
            print(res)
            log_str = self.get_log(res, "eval")
            print_rank("*" * 100)
            self.print_and_save(log_str)
            print_rank("*" * 100)

            ckpt_dir = os.path.join(self.args.save, f"{self.global_steps}")
            if res["spr"] > self.best_spr and os.path.exists(ckpt_dir):
                self.print_and_save(f"Unpdating best ckpt to {ckpt_dir}")
                self.best_ckpt_dir = ckpt_dir
                link_name = os.path.join(self.args.save, "best")
                link_name_tmp = os.path.join(self.args.save, "best_tmp")
                os.symlink(ckpt_dir, link_name_tmp)
                os.rename(link_name_tmp, link_name)
    
    def inference(self, save_path=None):
        save_path = save_path or self.args.save
        os.makedirs(save_path, exist_ok=True)
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False, drop_last=False, rank=self.dp_rank, num_replicas=self.dp_world_size)
        eval_dataloader = DataLoader(
            self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.eval_dataset.collate)
        
        if self.dp_rank == 0:
            check_indices = []
            tot = 64
            print("Creating sanity check indices.")
            while len(check_indices) < tot:
                x = random.randint(0, len(self.eval_dataset)-1)
                if x not in check_indices:
                    check_indices.append(x)
            check_indices = sorted(check_indices)
            torch.save(check_indices, os.path.join(save_path, f"check_indices_{numerize(self.min_offset)}_{numerize(len(self.eval_dataset))}.pt"))
            check_insts = []
            for n, cidx in enumerate(tqdm(check_indices)):
                print(f"{n}/{tot}")
                check_insts.append(self.eval_dataset[cidx][0].astype(int))
            torch.save(check_insts, os.path.join(save_path, f"check_insts_{numerize(self.min_offset)}_{numerize(len(self.eval_dataset))}.pt"))
            print("Sanity check indices created.")
        
        dist.barrier()
        self.model.eval()
        all_scores = []
        
        global_batch_size = self.args.eval_batch_size * self.dp_world_size
        num_per_shard = self.args.save_interval * global_batch_size
        if self.args.do_infer:
            log_str = f"Start from min_idx: {self.min_idx}, min_offset: {self.min_offset}, infer_num: {len(self.eval_dataset)}\n"
            log_str += f"Example num per shard: {num_per_shard}"
            self.print_and_save(log_str)

        st = time()
        idx = self.min_idx
        offset = self.min_offset
        with torch.no_grad():
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, f"Inference", disable=(not self.dp_rank == 0), file=sys.stdout)):
                if self.dp_rank == 0 and i == 0:
                    self.first_print(model_batch, no_model_batch)
                
                self.eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                score = self.model.inference(**model_batch)
                if self.args.torch_compile:
                    score = score.clone()
                all_scores.append(score)
                ct = time() - st
                if i % self.args.log_interval == 0:
                    print_rank("Infering {}. {}/{}. {}/{}. Spent time: {:.2f}".format(
                        idx,
                        i,
                        len(eval_dataloader),
                        self.min_offset + i*global_batch_size,
                        self.min_offset + len(self.eval_dataset),
                        ct
                    ))
                
                dist.barrier()
                if self.args.do_infer and (i+1) % self.args.save_interval == 0:
                    all_scores = torch.cat(all_scores, dim=0)
                    all_scores = all_gather(all_scores, dim=1, op="stack").view(-1)
                    all_scores = all_scores[:len(self.eval_dataset)-(idx-self.min_idx)*num_per_shard]
                    if self.dp_rank == 0:
                        state = {
                            "idx": idx+1, # next run start from this index
                            "offset": offset + len(all_scores)
                        }
                        torch.save(all_scores, os.path.join(save_path, f"scores_{idx}.pt"))
                        with open(os.path.join(save_path, "state.json"), "w") as f:
                            json.dump(state, f)
                        print(f"Saving {len(all_scores)} scores to", os.path.join(save_path, f"scores_{idx}.pt"))
                    dist.barrier()
                    idx += 1
                    offset += len(all_scores)
                    all_scores = []
        
        if len(all_scores) > 0:
            print_rank("Infering the last shard")
            all_scores = torch.cat(all_scores, dim=0)
            all_scores = all_gather(all_scores, dim=1, op="stack").view(-1)
            all_scores = all_scores[:len(self.eval_dataset)-(idx-self.min_idx)*num_per_shard]
        
            if self.dp_rank == 0:
                if self.args.do_infer:
                    state = {
                        "idx": idx+1,
                        "offset": offset + len(all_scores)
                    }
                    torch.save(all_scores, os.path.join(save_path, f"scores_{idx}.pt"))
                    with open(os.path.join(save_path, "state.json"), "w") as f:
                        json.dump(state, f)
                    print(f"Saving {len(all_scores)} scores to", os.path.join(save_path, f"scores_{idx}.pt"))
                else:
                    torch.save(all_scores, os.path.join(save_path, "scores.pt"))
                    print(f"Saving {len(all_scores)} scores to", os.path.join(save_path, f"scores.pt"))
                    all_scores = all_scores.cpu()
                    gold_scores = torch.tensor(self.eval_dataset.scores)
                    sorted_gold_scores, sorted_indices = torch.sort(gold_scores, descending=True)
                    sorted_scores = all_scores[sorted_indices]
                    plt.plot(sorted_gold_scores.numpy(), label="gold")
                    plt.scatter(range(len(sorted_scores)), sorted_scores.numpy(), label="pred", s=0.5, c="tab:orange")
                    plt.legend()
                    plt.savefig(os.path.join(save_path, "scores.png"))
                    plt.close()
                    
                    sp_corr = spearmanr(gold_scores, all_scores).statistic
                    acc, f1, rand_acc, rand_f1 = compute_acc(gold_scores, all_scores)                    
        
        dist.barrier()
        if not self.args.do_infer and self.dp_rank == 0:
            return {
                "acc": acc,
                "f1": f1,
                "spr": sp_corr,
                "rand_acc": rand_acc,
                "rand_f1": rand_f1
            }
        else:
            return {}


def compute_acc(gold_scores, scores):
    threshold = 0.3
    opt_labels = get_labels(gold_scores, threshold)
    preds = get_labels(scores, threshold)

    acc = accuracy_score(opt_labels, preds)
    f1 = f1_score(opt_labels, preds)
    
    rand_preds = [1 for _ in range(int(threshold * len(opt_labels)))] + [0 for _ in range(len(opt_labels) - int(threshold * len(opt_labels)))]
    random.shuffle(rand_preds)
    
    rand_acc = accuracy_score(opt_labels, rand_preds)
    rand_f1 = f1_score(opt_labels, rand_preds)
    
    return acc, f1, rand_acc, rand_f1


def get_labels(x, threshold):
    sorted_indices = torch.argsort(x, descending=True, stable=True)
    labels = torch.zeros_like(x)
    labels[:int(threshold * len(x))] = 1

    reodered_labels = torch.zeros_like(labels)
    for idx, l in zip(sorted_indices, labels):
        reodered_labels[idx] = l

    return reodered_labels.to(torch.int32).cpu().tolist()