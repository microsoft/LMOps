import os
import math
import random
import numpy as np
from tqdm import tqdm
from time import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.func import grad, jvp, vmap, grad_and_value

from transformers import AutoConfig, AutoModelForCausalLM

from utils import all_gather, print_rank, save_rank, get_model
from train_eval_utils import BaseTrainer
from data_utils import PromptDataset, LMDataset

from .model_wrapper import TransformerWrapper
from .checkpointing import Checkpointing
from .grad_utils import jvp_single, jvp_batch, hvp_fwdrev


class GammaTrainer(BaseTrainer):
    def __init__(self, args, device):
        super().__init__(args, None, device, do_train=True)
        self.dtype = torch.float32 if args.fp32 else torch.float16
        self.np_dtype = np.float32 if args.fp32 else np.float16
        # data        
        self.train_dataset, self.dev_dataset = self.get_data()
        self.prepare_learning(args)

        # model
        self.model = self.get_model(args, device)
        self.optimizer = self.get_optimizer(self.model, args)
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer, args)
        self.theta_size = self.model.get_params_vec().size()
        # self.thetas[i]: theta after the i-th update (before the i+1-th update)
        self.thetas = []
        
        # gamma
        self.lam = None
        self.grad_gamma = 0
        self.all_ct = []
        self.proxy_dataset = self.get_proxy_dataset()

        # checkpointing
        self.checkpointing = Checkpointing(
            self.args, self.tot_bs, self.total_steps, self.theta_size[-1], self.device, self.r)
        if self.r == 0:
            self.checkpointing.setup_h5(self.model)

        # dev/eval
        self.dev_data_cache = None
        self.dev_grad_first_printed = False
        self.eval_first_printed = False
                
        self.print_setup()

    def get_model(self, args, device):
        base_model = get_model(args, device)
        model = TransformerWrapper(base_model)
        return model

    def get_data(self, do_probe=True):
        train_dataset = LMDataset(self.args, self.tokenizer, f"data", self.args.data_dir, 
                                  self.args.train_num, do_probe=do_probe, min_offset=self.args.min_offset, max_state=self.args.max_state)
        # train_dataset = PromptDataset(self.args, self.tokenizer, f"train", self.args.data_dir, self.args.train_num, do_probe=do_probe)
        print_rank("train num", len(train_dataset))
        if self.args.dataset_type == "prompt":
            dev_dataset = PromptDataset(self.args, self.tokenizer, "data", self.args.dev_data_dir, self.args.dev_num, do_probe=do_probe)
        elif self.args.dataset_type == "lm":
            dev_dataset = LMDataset(self.args, self.tokenizer, "data", self.args.dev_data_dir, self.args.dev_num, do_probe=do_probe)
        else:
            raise NotImplementedError(f"Dataset type {self.args.dataset_type} not implemented")
        print_rank("valid num", len(dev_dataset))
        
        return train_dataset, dev_dataset

    def prepare_learning(self, args=None):
        args = args or self.args
        # various batch sizes
        self.r = self.dp_rank
        self.ws = self.dp_world_size
        self.bs = self.args.batch_size
        self.gacc = self.args.gradient_accumulation_steps
        self.px_bs = self.args.proxy_batch_size
        self.gl_bs = self.ws * self.bs
        self.tot_bs = self.bs * self.ws * self.gacc
        self.grad_bs = self.args.grad_batch_size or self.bs
        self.grad_gacc = self.tot_bs // (self.grad_bs * self.ws)

        self.train_iters_per_epoch = int(len(self.train_dataset) / self.tot_bs)
        assert (args.epochs is not None) ^ (
            args.total_iters is not None), (args.epochs, args.total_iters)
        self.total_steps = args.total_iters or self.train_iters_per_epoch * args.epochs
        self.epochs = args.epochs or math.ceil(
            args.total_iters / self.train_iters_per_epoch)

    def get_proxy_dataset(self):
        proxy_dataset = LMDataset(self.args, self.tokenizer, f"data", self.args.proxy_data_dir, self.args.proxy_num, do_probe=True)        
        self.print_and_save(f"proxy num {len(proxy_dataset)}")
        return proxy_dataset

    def print_setup(self):
        print_rank("grad_bs: {}, grad_gacc: {}".format(self.grad_bs, self.grad_gacc))
        self.checkpointing.print_setup()
        
    def evaluate(self, eval_dataset, cache=None, max_step=None, shuffle=False):
        if cache is None:
            eval_sampler = DistributedSampler(
                eval_dataset, shuffle=shuffle, drop_last=True, rank=self.r, num_replicas=self.ws)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers, collate_fn=eval_dataset.collate, drop_last=True)
            cache = []
            first_eval = True
        else:
            eval_dataloader = cache
            first_eval = False

        all_losses = []
        with torch.no_grad():
            self.model.eval()
            for i, (model_batch, no_model_batch) in enumerate(tqdm(eval_dataloader, disable=True)):
                if first_eval:
                    eval_dataset.move_to_device(model_batch, no_model_batch, self.device)
                    cache.append((model_batch, no_model_batch))
                if not self.eval_first_printed:
                    self.first_print(model_batch, no_model_batch)
                    self.eval_first_printed = True
                
                # forward
                loss, losses = self.model.compute_loss(**model_batch, **no_model_batch)
                all_losses.append(losses)

                if max_step is not None and i >= max_step:
                    break

            all_losses = torch.cat(all_losses, dim=0)
            all_losses = all_gather(all_losses)
            eval_loss = torch.mean(all_losses)
        
        return eval_loss, cache

    def forward(self):
        '''
        Forward with checkpointing
        '''
        train_sampler = DistributedSampler(
            self.train_dataset, shuffle=(not self.args.no_shuffle), drop_last=True, rank=self.r, num_replicas=self.ws)
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.bs,
            num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate, drop_last=True)

        self.steps = 0
        self.global_steps = 1
        self.epoch = 0
        
        full_batch_loss = 0

        _cached_data = []
        
        all_losses = []
        all_dev_losses = []
        pbar = tqdm(total=self.total_steps, desc="Train", disable=(self.r != 0))

        dev_loss, self.dev_data_cache = self.evaluate(self.dev_dataset, self.dev_data_cache)
        all_dev_losses.append(dev_loss.item())
        log_str = "Eval | Epoch: {} | Global Step: {} | Dev Loss: {:.4f}".format(
                self.epoch, self.global_steps, dev_loss)
        self.print_and_save(log_str)

        for e in range(self.epochs):
            self.epoch = e + 1

            for s, (model_batch, no_model_batch) in enumerate(train_dataloader):
                self.train_dataset.move_to_device(model_batch, no_model_batch, self.device)
                                                
                self.checkpointing.append_micro_step_data({**model_batch, **no_model_batch})

                if ((self.steps + 1) % self.gacc == 0) and (self.global_steps % self.args.eval_interval == 0):
                    dev_loss, self.dev_data_cache = self.evaluate(self.dev_dataset, self.dev_data_cache)
                    all_dev_losses.append(dev_loss.item())
                    log_str = "Eval | Epoch: {} | Global Step: {} | Dev Loss: {:.4f}".format(
                            self.epoch, self.global_steps, dev_loss)
                    self.print_and_save(log_str)

                # forward
                loss, _ = self.model.compute_loss(**model_batch, **no_model_batch)
                loss = loss / (self.ws * self.gacc)
                # backward
                loss.backward()
                
                # step
                if (self.steps + 1) % self.gacc == 0:
                    for param in self.model.parameters():
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.checkpointing.step()
                else:
                    step_time = 0

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                full_batch_loss += loss.item()
                
                # if self.global_steps % self.args.log_interval == 0:
                if (self.steps + 1) % self.gacc == 0 and (self.global_steps % self.args.log_interval == 0):
                    log_str = "Train | Epoch: {} | Global Step: {} | Loss: {:.4f}".format(
                        self.epoch, self.global_steps, full_batch_loss)
                    self.print_and_save(log_str)

                if (self.steps + 1) % self.gacc == 0:
                    all_losses.append(full_batch_loss)
                    full_batch_loss = 0.0
                
                # save
                if ((self.steps + 1) % self.gacc == 0) and self.checkpointing.do_dump_model(self.global_steps):
                    self.checkpointing.dump_model(self.model, self.global_steps)
                
                if ((self.steps + 1) % self.gacc == 0) and self.checkpointing.do_dump_data(self.global_steps):
                    self.checkpointing.dump_data(self.global_steps)

                # end
                if ((self.steps + 1) % self.gacc == 0) and (self.global_steps >= self.total_steps):
                    # self.save(self.args.save)
                    # self.evaluate()
                    self.checkpointing.dump_data(self.global_steps)

                    dev_area_loss = np.mean(all_dev_losses)
                    log_str = "Train | Epoch: {} Dev Area Loss: {:.4f}".format(self.epoch, dev_area_loss)
                    self.print_and_save(log_str)
                    pbar.close()

                    return all_losses, all_dev_losses
                
                # update steps
                self.steps += 1
                if self.steps % self.gacc == 0:
                    self.global_steps += 1
                    pbar.update(1)
        
        assert False, "Should not reach here"

    def inner_forward(self, start, end):
        '''
        Re-compute the forward pass from the checkpointed model parameters.
        '''
        if start % self.ws == self.r:
            self.thetas.append(self.model.get_params_vec())
        
        for i in range(start + 1, end):
            # the i-th update, use data with index i-1
            # i is the index of the model after the i-th update
            torch.cuda.synchronize()
            st = time()
            batch = self.checkpointing.load_h5_data(i-1)
            
            logging_loss = 0
            for ii in range(self.gacc):
                mini_batch = {k: v[ii*self.gl_bs:(ii+1)*self.gl_bs][self.r*self.bs:(self.r+1)*self.bs] for k, v in batch.items()}
                loss, _ = self.model.compute_loss(**mini_batch)
                loss = loss / (self.ws * self.gacc)
                loss.backward()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                logging_loss += loss.item()

            for param in self.model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if i % self.ws == self.r:
                self.thetas.append(self.model.get_params_vec())
            
            torch.cuda.synchronize()
            inner_fw_time = time() - st
            
            if i % self.args.log_interval == 0:
                log_str = "IF: Af. {}-th Forw. Upd., Loss: {:.4f}, IF_time: {:.2f}".format(i, logging_loss, inner_fw_time)
                print_rank(log_str)

    def compute_dev_grad(self, cache, params, buffers):
        steps = 0
        all_g_dev_vec = 0
        all_loss = 0
        for model_batch, no_model_batch in cache:
            if not self.dev_grad_first_printed:
                self.first_print(model_batch, no_model_batch)
                self.dev_grad_first_printed = True
            g_dev, loss = grad_and_value(self.model.compute_loss_func)(
                params, buffers, self.model, **model_batch, **no_model_batch)
            g_dev_vec = self.model.params_to_vector(g_dev)
            del g_dev
            dist.all_reduce(g_dev_vec, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            g_dev_vec = g_dev_vec / self.ws
            loss = loss / self.ws
            all_g_dev_vec += g_dev_vec
            all_loss += loss
            steps += 1
        
        all_g_dev_vec = all_g_dev_vec / steps
        all_loss = all_loss / steps
        
        return all_g_dev_vec, all_loss

    def inner_backward(self, start, end):
        '''
        Backward between two checkpointed models.
        '''
        for i in range(end-1, start-1, -1):
            # i is the index of the model before the i-th update in forward
            # \lambda_i = \nabla L_{dev}(\theta_i) + \lambda_{i+1} - lr * \nabla^2 \sum_n \gamma_{i,n} L(\theta_i, x_{i,n}) \lambda_{i+1}
            torch.cuda.synchronize()
            st0 = time()
            if i % self.ws == self.r:
                theta = self.thetas.pop()
            else:
                theta = torch.zeros(self.theta_size, device=self.device, dtype=self.dtype)
            dist.broadcast(theta, i % self.ws)
                        
            params = self.model.vector_to_params(theta)
            buffers = {n: b.detach() for n, b in self.model.named_buffers()}

            st = time()
            torch.cuda.synchronize()
            # 1. \partial L_{dev} / \partial \theta_{t-1}   
            g_dev_vec, dev_loss = self.compute_dev_grad(self.dev_data_cache, params, buffers)
                
            torch.cuda.synchronize()
            dev_grad_time = time() - st

            if self.lam is None:
                self.lam = g_dev_vec
                log_str = "IB: Bef. {}-th Forw. Upd., Lam norm: {:.6f}, Dev gnorm: {:.4f}, Dev loss: {:.4f}".format(
                    i, self.lam.norm().item(), g_dev_vec.norm().item(), dev_loss.item())
                self.print_and_save(log_str)
                continue
            
            batch = self.checkpointing.load_h5_data(i)
            
            batch = {k:v[self.r*self.gacc*self.bs:(self.r+1)*self.gacc*self.bs] for k, v in batch.items()}

            st = time()
            torch.cuda.synchronize()

            # 2. \partial L / \partial \gamma_t
            lam_param = self.model.vector_to_params(self.lam)

            if self.args.compute_ct_interval is None or i % self.args.compute_ct_interval == 0:
                ct = self.compute_proxy_ct(lam_param, params, buffers)
            else:
                ct = torch.zeros(len(self.proxy_dataset), device=self.device, dtype=self.dtype)

            torch.cuda.synchronize()
            samp_grad_time = time() - st

            st = time()
            torch.cuda.synchronize()
            # 3. \partial L / \partial \theta_{t} @ \partial \theta_{t} / \partial \theta_{t-1}
            hvp_vec = hvp_fwdrev(
                self.model, batch, lam_param, params, buffers, self.grad_bs, self.grad_gacc, self.ws)
            dist.all_reduce(hvp_vec, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            hess_time = time() - st

            # 4. update lam
            tmp = self.args.lr * hvp_vec
            self.lam = g_dev_vec + self.lam - tmp
            
            # 5. step
            self.step_in_backward(ct, i)
            torch.cuda.synchronize()
            inner_bw_time = time() - st0
            if i % self.args.log_interval == 0:
                log_str = "IB: Bef. {}-th Forw. Upd., l_dev: {:.4f}, nL: {:.3g}, nDG: {:.3g}, nTMP: {:.3g}, tIB: {:.2f}, tDG: {:.2f}, tSG: {:.2f}, tHS: {:.2f}".format(
                    i, dev_loss.item(), self.lam.norm().item(), g_dev_vec.norm().item(), tmp.norm().item(), inner_bw_time, dev_grad_time, samp_grad_time, hess_time)
                self.print_and_save(log_str)

    def compute_proxy_ct(self, lam_param, params, buffers):
        '''
        Compute the contribution of the proxy data to the gradient of the loss w.r.t. the model parameters.
        '''
        proxy_sampler = DistributedSampler(
            self.proxy_dataset, shuffle=False, drop_last=False, rank=self.r, num_replicas=self.ws)
        proxy_dataloader = DataLoader(
            self.proxy_dataset, sampler=proxy_sampler, batch_size=self.px_bs,
            num_workers=self.args.num_workers, collate_fn=self.train_dataset.collate, drop_last=False)
        
        all_ct = []
        for i, (model_batch, no_model_batch) in enumerate(proxy_dataloader):
            if i % 10 == 0:
                print_rank("Computing contribution on proxy data: {}/{}".format(i, len(proxy_dataloader)))
            self.proxy_dataset.move_to_device(model_batch, no_model_batch, self.device)
            batch = {**model_batch, **no_model_batch}
            ct = jvp_batch(self.model, batch, lam_param, params, buffers)
            ct = all_gather(ct, dim=1, op="stack").view(-1)
            ct = self.args.lr * ct
            all_ct.append(ct)
        all_ct = torch.cat(all_ct, dim=0)
        all_ct = all_ct[:len(self.proxy_dataset)]
        
        return all_ct

    def step_in_backward(self, ct, t):
        self.grad_gamma = self.grad_gamma + ct
        if t == 0 and self.r == 0:
            torch.save(self.grad_gamma, os.path.join(self.args.save, "grad_gamma.pt"))

    def backward(self):
        '''
        Backward with checkpointing
        '''
        for s in range(self.checkpointing.ckpt_num, -1, -1):
            # s is the start checkpoint
            if self.r == 0:
                v = self.checkpointing.load_h5_model(s)
            else:
                v = torch.zeros(self.theta_size, device=self.device, dtype=self.dtype)
            dist.broadcast(v, 0)
            self.model.set_params_vec(v)
            start = s * self.checkpointing.dump_model_interval
            end = (s+1) * self.checkpointing.dump_model_interval
            end = min(end, self.total_steps + 1)

            self.inner_forward(start, end)
            self.inner_backward(start, end)
            assert len(self.thetas) == 0
        
        dist.barrier()
        
        self.lam = None

    def train(self):
        '''
        Solve gamma for only one epoch.
        '''
        fw_time = time()
        torch.cuda.synchronize()
        
        all_losses, all_dev_losses = self.forward()
        interval = 10
        sub_all_losses = [round(all_losses[i], 4) for i in range(0, len(all_losses), interval)]
        sub_all_dev_losses = [round(all_dev_losses[i], 4) for i in range(0, len(all_dev_losses), interval)]
        self.print_and_save(f"Train Losses (inter {interval}): {sub_all_losses}")
        self.print_and_save(f"Dev Losses (inter {interval}): {sub_all_dev_losses}")

        torch.cuda.synchronize()
        fw_time = time() - fw_time
        bw_time = time()
        
        self.backward()
        
        torch.cuda.synchronize()
        bw_time = time() - bw_time
        
        self.print_and_save("Forward Time: {:.4f}, Backward Time: {:.4f}".format(fw_time, bw_time))
