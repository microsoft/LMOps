import torch
from torch.optim import SGD
import numpy as np
import os
import wandb
import time
from torch.func import grad, vmap
from utils import save_rank, print_rank, all_gather
import torch.distributed as dist

from transformers import get_constant_schedule_with_warmup


class BaseTrainer():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        self.exp_name = args.save.strip("/").replace(args.base_path.strip("/"), "").replace("_", "").replace("/", "_").strip("_")
        
        self.tokenizer = self.get_tokenizer()
        if self.tokenizer is not None:
            print_rank("vocab size: {}".format(self.tokenizer.vocab_size))
        self.max_length = args.max_length
        print_rank("max length: {}".format(self.max_length))

        self.model_init_dir = os.path.join(args.base_path, "data", args.data_names, "model_init")
        os.makedirs(self.model_init_dir, exist_ok=True)
        model_init_path = os.path.join(self.model_init_dir, f"{args.model_name}-new.pt")
        
        self.model = self.get_model()
        print_rank(' > number of parameters: {}'.format(
            sum([p.nelement() for p in self.model.parameters()])), flush=True)
        
        if args.load_cache is not None:
            if not os.path.exists(model_init_path):
                if dist.get_rank() == 0:
                    torch.save(self.model.state_dict(), model_init_path)

            dist.barrier()
            self.model.load_state_dict(torch.load(model_init_path, map_location="cpu"))
        
        self.optimizer = SGD(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_iters)
    
        self.train_data, self.dev_data, self.test_data = self.get_data()
        self.train_data = self.reform_data(self.train_data)
        self.dev_data = self.reform_data(self.dev_data)
        self.test_data = self.reform_data(self.test_data)

        print_rank("train data size: {} | dev data size: {} | test data size: {}".format(
            (self.train_data[0].size(), self.train_data[1].size()), 
            (self.dev_data[0].size(), self.dev_data[1].size()), 
            (self.test_data[0].size(), self.test_data[1].size())))

    def reload_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_init_dir, f"{self.args.model_name}-new.pt")))
    
    def get_tokenizer(self):
        raise NotImplementedError
    
    def get_model(self):
        raise NotImplementedError

    def reform_data(self):
        raise NotImplementedError

    def get_data(self):
        all_data_splits = {}
        for split in ["train", "dev", "test"]:
            data = torch.load(os.path.join(self.args.data_dir, f"{split}.pt"))
            all_data_splits[split] = data
        all_data_splits["train"] = all_data_splits["train"][:self.args.train_num]
        all_data_splits["dev"] = all_data_splits["dev"][:self.args.dev_num]
        all_data_splits["test"] = all_data_splits["test"][:self.args.dev_num]
        return all_data_splits["train"], all_data_splits["dev"], all_data_splits["test"]
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def calc_grad_eval(self, eval_xn, eval_yn):
        with torch.no_grad():
            params = {n: p.detach() for n, p in self.model.named_parameters()}
            buffers = {n: p.detach() for n, p in self.model.named_buffers()}
            
            r = dist.get_rank()
            grad_eval_func = grad(self.model.compute_loss_func)
            
            eval_bs = self.args.eval_batch_size
            gl_eval_bs = dist.get_world_size() * eval_bs
            eval_grad_acc_steps = eval_xn.size(0) // gl_eval_bs
            grad_eval_vec = 0
            for i in range(eval_grad_acc_steps):
                eval_xn_batch = eval_xn[i*gl_eval_bs:(i+1)*gl_eval_bs][r*eval_bs:(r+1)*eval_bs]
                eval_yn_batch = eval_yn[i*gl_eval_bs:(i+1)*gl_eval_bs][r*eval_bs:(r+1)*eval_bs]
                grad_eval_batch = grad_eval_func(params, buffers, self.model, eval_xn_batch, eval_yn_batch)
                grad_eval_vec += self.model.params_to_vector(grad_eval_batch)
            
            dist.all_reduce(grad_eval_vec, op=dist.ReduceOp.SUM)
            grad_eval_vec /= (eval_grad_acc_steps * dist.get_world_size())
        
        return grad_eval_vec

    def calc_CT(self, xn, yn, eval_xn, eval_yn, gamma=None):
        with torch.no_grad():
            params = {n: p.detach() for n, p in self.model.named_parameters()}
            buffers = {n: p.detach() for n, p in self.model.named_buffers()}
            
            r = dist.get_rank()
            
            grad_eval_vec = self.calc_grad_eval(eval_xn, eval_yn)
            grad_eval = self.model.vector_to_params(grad_eval_vec)
            
            grad_train_single_func = grad(self.model.compute_loss_func_single)
            grad_train_func = vmap(grad_train_single_func, in_dims=(None, None, None, 0, 0))
            
            grad_bs = self.args.grad_batch_size
            gl_grad_bs = dist.get_world_size() * grad_bs
            grad_acc_steps = xn.size(0) // gl_grad_bs
            
            CT = torch.zeros(xn.size(0), device=self.device)

            for i in range(grad_acc_steps):
                xn_batch = xn[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs]
                yn_batch = yn[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs]
                grad_train = grad_train_func(params, buffers, self.model, xn_batch, yn_batch)
            
                for n, _ in self.model.named_parameters():
                    x1 = grad_eval[n].view(-1)
                    x2 = grad_train[n].contiguous().view(grad_train[n].size(0), -1)
                    CT[i*gl_grad_bs:(i+1)*gl_grad_bs][r*grad_bs:(r+1)*grad_bs] += x2 @ x1
            
            dist.all_reduce(CT, op=dist.ReduceOp.SUM)
            
        return CT

    def evaluate(self, eval_data):
        losses = []
        
        r = dist.get_rank()
        eval_bs = self.args.eval_batch_size
        gl_eval_bs = dist.get_world_size() * eval_bs
        grad_acc_steps = eval_data[0].size(0) // gl_eval_bs
        with torch.no_grad():
            for i in range(grad_acc_steps):
                input_batch = eval_data[0][i*gl_eval_bs:(i+1)*gl_eval_bs][r*eval_bs:(r+1)*eval_bs]
                label_batch = eval_data[1][i*gl_eval_bs:(i+1)*gl_eval_bs][r*eval_bs:(r+1)*eval_bs]                
                loss, _ = self.model.compute_loss(input_batch, label_batch)
                losses.append(loss.item())

        loss = torch.tensor(np.sum(losses)).to(self.device)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()

        return loss

    def train(self, gamma=None, wandb_name="baseline", calc_CT=False):
        save_path = os.path.join(self.args.save, wandb_name)
        os.makedirs(save_path, exist_ok=True)

        if dist.get_rank() == 0:
            run = wandb.init(
                name=f"{wandb_name}",
                project="toy-trm",
                group=self.exp_name,
                config=self.args,
                reinit=True,
                tags=[self.args.time_stamp, self.args.data_names])
        
        st = time.time()
        all_dev_loss, all_test_loss = [], []
        all_dev_CT, all_test_CT = [], []
        all_train_losses_per_inst = []
        
        if self.args.batch_size == -1:
            self.args.batch_size = self.train_data[0].size(0)
        if self.args.eval_batch_size == -1:
            self.args.eval_batch_size = self.dev_data[0].size(0)
        
        bs = self.args.batch_size
        gl_bs = dist.get_world_size() * self.args.batch_size
        assert self.train_data[0].size(0) % gl_bs == 0, (self.train_data[0].size(0), self.args.batch_size, dist.get_world_size())
        eval_bs = self.args.eval_batch_size
        gl_eval_bs = dist.get_world_size() * self.args.eval_batch_size
        assert self.dev_data[0].size(0) % gl_eval_bs == 0, (self.dev_data[0].size(0), self.args.eval_batch_size, dist.get_world_size())
        
        min_dev_loss = 1e8
        min_dev_loss_epoch = -1
        
        r = dist.get_rank()
        
        if gamma is None:
            flat_gamma = torch.ones(self.train_data[0].size(0), device=self.device) / self.train_data[0].size(0)
        
        grad_acc_steps = self.train_data[0].size(0) // gl_bs
        for e in range(self.args.epochs):
            epoch_st = time.time()
            self.optimizer.zero_grad()
            gamma_e = gamma[e] if gamma is not None else flat_gamma
            train_losses, train_losses_per_inst = [], []
            dev_loss = self.evaluate(self.dev_data)
            test_loss = self.evaluate(self.test_data)

            # params = {n: p.detach() for n, p in self.model.named_parameters()}
            # buffers = {n: p.detach() for n, p in self.model.named_buffers()}
            # g_params = {n:0 for n, _ in self.model.named_parameters()}
            
            for i in range(grad_acc_steps):
                if e == 0 and i == 0:
                    print_rank(self.train_data[0][0])
                    if self.tokenizer is not None:
                        print_rank(self.tokenizer.decode(self.train_data[0][0].cpu().tolist()))
                    print_rank()
                
                train_input_batch = self.train_data[0][i*gl_bs:(i+1)*gl_bs][r*bs:(r+1)*bs]
                train_label_batch = self.train_data[1][i*gl_bs:(i+1)*gl_bs][r*bs:(r+1)*bs]
                gamma_e_batch = gamma_e[i*gl_bs:(i+1)*gl_bs][r*bs:(r+1)*bs]
                
                loss, losses = self.model.compute_loss(train_input_batch, train_label_batch, gamma=gamma_e_batch)
                
                loss.backward()
                
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                losses = all_gather(losses)
                
                # g, loss = grad_and_value(self.model.compute_loss_func)(params, buffers, self.model, self.train_data[0][start:end], self.train_data[1][start:end], gamma=batch_gamma_e)
                # for k in g_params.keys():
                #     g_params[k] += g[k]
                
                train_losses.append(loss.item())
                train_losses_per_inst.append(losses)
            
            for param in self.model.parameters():
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            
            # for n, p in self.model.named_parameters():
            #     p.data.add_(g_params[n], gamma=-self.args.lr)

            loss = np.sum(train_losses)
            train_losses_per_inst = torch.cat(train_losses_per_inst, dim=0)

            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            gn = self.get_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

            all_dev_loss.append(dev_loss)
            all_test_loss.append(test_loss)
            all_train_losses_per_inst.append(train_losses_per_inst)
            
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                min_dev_loss_epoch = e
                        
            if calc_CT:
                dev_CT = self.calc_CT(*self.train_data, *self.dev_data, gamma=gamma_e)
                all_dev_CT.append(dev_CT)
                
                test_CT = self.calc_CT(*self.train_data, *self.test_data, gamma=gamma_e)
                all_test_CT.append(test_CT)
            
            if dist.get_rank() == 0:
                wandb_log = {
                    "train_loss": loss.item(),
                    "dev_loss": dev_loss.item(),
                    "test_loss": test_loss.item(),
                    "grad_norm": gn,

                }
                
                wandb.log(wandb_log)
                
                if e % self.args.log_interval == 0:
                    log_str = "epoch {} | train loss {:.4f} | dev loss {:.4f} | test loss {:.4f} | gn: {:.4f} | lr:{:.4e} | single epoch time: {}\n".format(
                        e, loss.item(), dev_loss.item(), test_loss.item(), gn, self.lr_scheduler.get_last_lr()[0], time.time() - epoch_st)

                    print(log_str)
                    save_rank(log_str, os.path.join(save_path, "log.txt"))
        
        final_dev_loss = self.evaluate(self.dev_data)
        final_test_loss = self.evaluate(self.test_data)
        
        if dist.get_rank() == 0:
            print("all_time", time.time() - st)
            log_str = "min dev loss epoch: {} | min dev loss: {:.4f} | test loss: {:.4f}\n".format(min_dev_loss_epoch, min_dev_loss, all_test_loss[min_dev_loss_epoch])
            print(log_str)
            save_rank(log_str, os.path.join(save_path, "log.txt"))
            
            print("final | dev loss {:.4f} | test loss {:.4f}".format(final_dev_loss.item(), final_test_loss.item()))
            
            torch.save((all_dev_loss, all_test_loss), os.path.join(save_path, "all_loss.pt"))
            torch.save(torch.stack(all_train_losses_per_inst, dim=0), os.path.join(save_path, "all_train_losses_per_inst.pt"))
            if calc_CT:
                torch.save(all_dev_CT, os.path.join(save_path, "all_dev_CT.pt"))
                torch.save(all_test_CT, os.path.join(save_path, "all_test_CT.pt"))
            
            run.finish()
        
        dist.barrier()
            