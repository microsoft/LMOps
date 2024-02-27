import torch
import torch.nn as nn
import cvxpy as cp
import os
import time
from tqdm import tqdm
from utils import print_rank, save_rank
import torch.distributed as dist
from torch.func import grad, vmap, grad_and_value, jvp
from torch.multiprocessing import Pool
from transformers import get_linear_schedule_with_warmup

from perceptron_trainer import PerceptronTrainer
from transformer_trainer import TransformerTrainer


class ProjSolver():
    def __init__(self):
        pass
    
    def solve(self, lid_data):
        lid, data = lid_data
        data_cpu = data.squeeze().numpy()
        data_proj = cp.Variable(data.size(0))
        objective = cp.Minimize(cp.sum_squares(data_cpu - data_proj))
        prob = cp.Problem(objective, [cp.sum(data_proj) == 1, data_proj >= 0])
        prob.solve()
        data_res = torch.tensor(data_proj.value).view(data.size()).to(data.dtype)
        return lid, data_res


def proj_gamma(optimizer, args, kwargs):
    
    all_gammas = torch.stack([torch.zeros_like(p.data) for p in optimizer.param_groups[0]["params"]], dim=0)
    
    if dist.get_rank() == 0:
        solver = ProjSolver()

        pool = Pool(processes=20)
        all_data = [p.data.cpu() for p in optimizer.param_groups[0]["params"]]
        solved_gamma = pool.map(solver.solve, enumerate(all_data))
        with tqdm(total=len(all_gammas), desc="Proj Alpha") as pbar:
            for lid, data_res in solved_gamma:
                all_gammas[lid] = data_res.to(all_gammas.device)
                pbar.update(1)
        
    dist.broadcast(all_gammas, 0)
    for lid, p in enumerate(optimizer.param_groups[0]["params"]):
        p.data = all_gammas[lid]        


max_grad = 0
min_grad = 1e6
max_grad_epoch = 0
min_grad_epoch = 0


class GradLayerFunction(torch.autograd.Function):   
    @staticmethod
    def clip_grad(theta, max_norm):
        if max_norm < 0:
            return theta, torch.tensor(1.0, device=theta.device)
        total_norm = torch.norm(theta)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        theta.mul_(clip_coef_clamped)
        return theta, clip_coef_clamped
     
    @staticmethod
    def forward(ctx, theta, gamma, pn_gamma, model, xn, yn, dev_xn, dev_yn, eta, t, args):
        
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}

        r = dist.get_rank()
        
        ctx.save_for_backward(xn, yn, dev_xn, dev_yn)
        if args.toy_zero2:
            if t % dist.get_world_size() == r:
                ctx.theta = theta
            else:
                ctx.theta = None
            ctx.theta_size = theta.size()
        else:
            ctx.theta = theta

        ctx.model = model
        ctx.gamma = gamma
        ctx.pn_gamma = pn_gamma
        ctx.eta = eta
        ctx.t = t
        ctx.args = args
        
        
        # NOTE: compute dev loss at the beginning of each step
        eval_bs = args.eval_batch_size
        gl_eval_bs = dist.get_world_size() * eval_bs
        dev_grad_acc_steps = dev_xn.size(0) // gl_eval_bs
        losses = 0
        for i in range(dev_grad_acc_steps):
            dev_xn_batch = (dev_xn[i*gl_eval_bs:(i+1)*gl_eval_bs])[r*eval_bs:(r+1)*eval_bs]
            dev_yn_batch = (dev_yn[i*gl_eval_bs:(i+1)*gl_eval_bs])[r*eval_bs:(r+1)*eval_bs]
            loss = model.compute_loss_func(params, buffers, model, dev_xn_batch, dev_yn_batch)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
            losses += loss
        dev_loss = losses / dev_grad_acc_steps
        
        if gamma is None:
            return dev_loss, None
        
        bs = args.batch_size
        gl_bs = dist.get_world_size() * bs
        grad_acc_steps = xn.size(0) // gl_bs
        g_vec = 0
        for i in range(grad_acc_steps):
            xn_batch = (xn[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
            yn_batch = (yn[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
            gamma_batch = (gamma[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
            g, _ = grad_and_value(model.compute_loss_func)(params, buffers, model, xn_batch, yn_batch, gamma=gamma_batch)
            g_vec_b = model.params_to_vector(g)
            dist.all_reduce(g_vec_b, op=dist.ReduceOp.SUM)
            g_vec += g_vec_b
        
        g_params = model.vector_to_params(g_vec)
        
        g_vec_clipped, clip_coef = GradLayerFunction.clip_grad(g_vec, args.clip_grad)        
        g_params_clip = model.vector_to_params(g_vec_clipped)

        new_theta = theta.clone()
        ctx.clip_coef = clip_coef
        new_theta.add_(g_vec, alpha=-eta)

        return dev_loss, new_theta

    @staticmethod
    def backward(ctx, loss_grad_output, grad_output):
        # print(ctx.t)
        if ctx.t % 100 == 0:
            print_rank("Backward", ctx.t, ctx.eta)

        xn, yn, dev_xn, dev_yn = ctx.saved_tensors
        gamma = ctx.gamma
        model = ctx.model
        eta = ctx.eta
        args = ctx.args

        r = dist.get_rank()
        
        if args.toy_zero2:
            if ctx.t % dist.get_world_size() == r:
                theta = ctx.theta
            else:
                theta = torch.zeros(ctx.theta_size, device=xn.device)
            dist.broadcast(theta, ctx.t % dist.get_world_size())
        else:
            theta = ctx.theta
        
        params = model.vector_to_params(theta)
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        
        # 1. \partial L_{dev} / \partial \theta_{t-1}
        eval_bs = args.eval_batch_size
        gl_eval_bs = dist.get_world_size() * eval_bs
        dev_grad_acc_steps = dev_xn.size(0) // gl_eval_bs
        g_dev_vec = 0
        for i in range(dev_grad_acc_steps):
            dev_xn_batch = (dev_xn[i*gl_eval_bs:(i+1)*gl_eval_bs])[r*eval_bs:(r+1)*eval_bs]
            dev_yn_batch = (dev_yn[i*gl_eval_bs:(i+1)*gl_eval_bs])[r*eval_bs:(r+1)*eval_bs]
            # print_rank("dev_xn_batch", dev_xn_batch.size())
            g_dev = grad(ctx.model.compute_loss_func)(params, buffers, model, dev_xn_batch, dev_yn_batch, None)
            # print_rank(g_dev["base_model.model.layers.0.self_attn.q_proj.weight"])
            # print_rank(g_dev["base_model.model.layers.0.self_attn.q_proj.weight"], rank=1)
            g_dev_b = ctx.model.params_to_vector(g_dev)
            # print_rank("g_dev_b", g_dev_b)
            # print_rank("g_dev_b", g_dev_b, rank=1)
            dist.all_reduce(g_dev_b, op=dist.ReduceOp.SUM)
            g_dev_vec += g_dev_b
            del g_dev
        g_dev_vec = g_dev_vec / (dev_grad_acc_steps * dist.get_world_size())
        # print_rank("g_dev", g_dev_vec)

        g_dev_vec = g_dev_vec * loss_grad_output
        
        # print_rank("g_dev", g_dev_vec)
        
        grad_theta = g_dev_vec
        
        if gamma is None:
            # last step
            return grad_theta, None, None, None, None, None, None, None, None, None, None
        
        # print_rank("grad_out", grad_output)
        
        # print_rank("g_dev", g_dev_vec)
        
        # not last step
        grad_output_params = model.vector_to_params(grad_output)
        
        # 2. \partial L / \partial \gamma_t
        vmapped_grad_func = vmap(grad(model.compute_loss_func_single), in_dims=(None, None, None, 0, 0))
        grad_bs = args.grad_batch_size
        gl_grad_bs = dist.get_world_size() * grad_bs
        grad_acc_steps_sample = xn.size(0) // gl_grad_bs
        CT_abs = torch.zeros_like(gamma)
        
        max_sample_grad_norm = 0
        
        for i in range(grad_acc_steps_sample):
            xn_batch = (xn[i*gl_grad_bs:(i+1)*gl_grad_bs])[r*grad_bs:(r+1)*grad_bs]
            yn_batch = (yn[i*gl_grad_bs:(i+1)*gl_grad_bs])[r*grad_bs:(r+1)*grad_bs]
            vmapped_g = vmapped_grad_func(params, buffers, model, xn_batch, yn_batch)
            for n, _ in model.named_parameters():
                x1 = grad_output_params[n].view(-1)
                x2 = vmapped_g[n].contiguous().view(vmapped_g[n].size(0), -1)
                max_sample_grad_norm = max(max_sample_grad_norm, torch.norm(x2, dim=1).max().item())
                (CT_abs[i*gl_grad_bs:(i+1)*gl_grad_bs])[r*grad_bs:(r+1)*grad_bs] += x2 @ x1
        
        dist.all_reduce(CT_abs, op=dist.ReduceOp.SUM)
        max_sample_grad_norm = torch.tensor(max_sample_grad_norm).to(CT_abs.device)
        dist.all_reduce(max_sample_grad_norm, op=dist.ReduceOp.MAX)
        max_sample_grad_norm = max_sample_grad_norm.item()
        
        grad_gamma = -ctx.clip_coef * CT_abs * eta
        
        # 3. \partial L / \partial \theta_{t} @ \partial \theta_{t} / \partial \theta_{t-1}
        bs = args.batch_size
        gl_bs = dist.get_world_size() * bs
        grad_acc_steps = xn.size(0) // gl_bs
        def hvp_fwdrev(f, primals, tangents):
            def grad_wrapper(pr):
                g = {n: 0 for n in params}
                for i in range(grad_acc_steps):
                    xn_batch = (xn[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
                    yn_batch = (yn[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
                    gamma_batch = (gamma[i*gl_bs:(i+1)*gl_bs])[r*bs:(r+1)*bs]
                    _g = grad(f)(pr, buffers, model, xn_batch, yn_batch, gamma=gamma_batch)
                    for n in g:
                        g[n] += _g[n]
                return g
            return jvp(grad_wrapper, primals, tangents)[1]
        
        # def hvp_revrev(f, primals, tangents):
        #     def grad_wrapper(pr):
        #         g = grad(f)(pr, buffers, model, xn, yn, gamma=gamma)
        #         return g
        #     vjpfunc = vjp(grad_wrapper, primals[0])[1]
        #     return vjpfunc(tangents[0])[0]
        
        hvp = hvp_fwdrev(model.compute_loss_func, (params,), (grad_output_params,))
        
        hvp_vec = model.params_to_vector(hvp)
        
        dist.all_reduce(hvp_vec, op=dist.ReduceOp.SUM)
        
        # print_rank("hvp", hvp_vec)
        # exit(0)
        
        grad_theta = grad_theta + (grad_output - ctx.clip_coef * eta * hvp_vec)

        # TODO: more accurate way to compute the gradient of gamma with clip_coef
        global max_grad, min_grad, max_grad_epoch, min_grad_epoch
        if torch.max(grad_gamma).item() > max_grad:
            max_grad = torch.max(grad_gamma).item()
            max_grad_epoch = ctx.t
        if torch.min(grad_gamma).item() < min_grad:
            min_grad = torch.min(grad_gamma).item()
            min_grad_epoch = ctx.t

        log_str = "{} {:.4e} grad out norm {:.4f} max sample grad norm {:.4f} dev vec norm {:.4f} grad norm {:.4f} max_grad {:.6f} min_grad {:.6f}".format(
            ctx.t, ctx.eta,
            torch.norm(grad_output).item(), max_sample_grad_norm, torch.norm(g_dev_vec).item(), 
            torch.norm(grad_gamma).item(), torch.max(grad_gamma).item(), torch.min(grad_gamma).item()
        )
        # print_rank(log_str)
        # exit(0)
        if ctx.t % 100 == 0:
            save_rank(log_str, os.path.join(args.save, "grad_log.txt"))

        return grad_theta, grad_gamma, None, None, None, None, None, None, None, None, None


def constant_schedule_with_warmup(lr, n_wm_steps, t):
    if t < n_wm_steps:
        return lr * t / n_wm_steps
    else:
        return lr


class GammaModel(nn.Module):
    def __init__(self, args, n_gamma, n_steps, n_wm_steps) -> None:
        super().__init__()
        self.args = args
        self.n_gamma = n_gamma
        self.n_steps = n_steps
        self.n_wm_steps = n_wm_steps
        self.gamma = nn.ParameterList(
            [nn.Parameter(torch.ones(n_gamma) / n_gamma) for _ in range(n_steps)])
        
    def forward(self, theta, model, xn, yn, dev_xn, dev_yn, eta, mode="dev"):
        all_losses, all_logging_losses = [], []
        area_loss = 0
        st = time.time()
        
        inner_log_interval = 10
        for t in tqdm(range(self.n_steps), desc=f"{mode} forward", disable=(dist.get_rank() != 0)):
            cur_eta = constant_schedule_with_warmup(eta, self.args.warmup_iters, t)
            prev_gamma = self.gamma[t-1] if t > 0 else None
            next_gamma = self.gamma[t+1] if t < self.n_steps - 1 else None
            pn_gamma = (prev_gamma, next_gamma)
            if t < self.n_wm_steps:
                with torch.no_grad():
                    loss, theta = GradLayerFunction.apply(
                        theta, self.gamma[t], pn_gamma, model, xn, yn, dev_xn, dev_yn, cur_eta, t, self.args)
            else:
                loss, theta = GradLayerFunction.apply(
                    theta, self.gamma[t], pn_gamma, model, xn, yn, dev_xn, dev_yn, cur_eta, t, self.args)

            if t % inner_log_interval == 0:
                # print("Forward | t: {} | inner loss: {:.4f}".format(t, loss.item()))
                all_logging_losses.append(round(loss.item(), 4))
        
            all_losses.append(loss.item())
            area_loss += loss
        
        loss, _ = GradLayerFunction.apply(
            theta, None, pn_gamma, model, xn, yn, dev_xn, dev_yn, eta, self.n_steps, self.args)
        area_loss += loss
        all_losses.append(loss.item())

        area_loss = area_loss / self.n_steps
        return area_loss, all_losses, all_logging_losses
    
    def get_trainable_params(self):
        trainable_params = []
        for n, p in self.named_parameters():
            n = n.split(".")
            if int(n[1]) >= self.n_wm_steps:
                trainable_params.append(p)
        return trainable_params

    
class OptPolicyTrainer():
    def __init__(self, args, device) -> None:
        
        if args.data_names == "tinystory":
            base_trainer_cls = TransformerTrainer
        elif args.data_names == "linear":
            base_trainer_cls = PerceptronTrainer
        else:
            raise NotImplementedError(args.data_names)

        self.base_trainer = base_trainer_cls(args, device)
        
        self.model = self.base_trainer.model
        self.train_data = self.base_trainer.train_data
        self.dev_data = self.base_trainer.dev_data
        self.test_data = self.base_trainer.test_data
        self.args = args
        self.device = device

        if self.args.batch_size == -1:
            self.args.batch_size = self.train_data[0].size(0)
        if self.args.eval_batch_size == -1:
            self.args.eval_batch_size = self.dev_data[0].size(0)
        if self.args.grad_batch_size == -1:
            self.args.grad_batch_size = self.train_data[0].size(0)

        gl_bs = dist.get_world_size() * self.args.batch_size
        assert self.train_data[0].size(0) % gl_bs == 0, (self.train_data[0].size(0), self.args.batch_size, dist.get_world_size())
        gl_eval_bs = dist.get_world_size() * self.args.eval_batch_size
        assert self.dev_data[0].size(0) % gl_eval_bs == 0, (self.dev_data[0].size(0), self.args.eval_batch_size, dist.get_world_size())
        gl_grad_bs = dist.get_world_size() * self.args.grad_batch_size
        assert self.train_data[0].size(0) % gl_grad_bs == 0, (self.train_data[0].size(0), self.args.grad_batch_size, dist.get_world_size())
        
        self.outer_epochs = args.outer_epochs
        self.outer_lr = args.outer_lr
        self.gamma_model = GammaModel(args, self.train_data[0].size(0), args.epochs, args.opt_gamma_wm_steps).to(device)
        self.optimizer = torch.optim.SGD(self.gamma_model.get_trainable_params(), lr=self.outer_lr)
        self.optimizer.register_step_post_hook(proj_gamma)
        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 0)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0, self.outer_epochs)
    
    def train(self, wandb_name=None):
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        theta = self.model.params_to_vector(params)
        xn, yn = self.train_data
        dev_xn, dev_yn = self.dev_data
        test_xn, test_yn = self.test_data
        for e in range(self.outer_epochs):
            save_rank("Epoch {}".format(e), os.path.join(self.args.save, "grad_log.txt"))
            st = time.time()
            self.optimizer.zero_grad()
            area_loss, all_losses, all_logging_losses = self.gamma_model(
                theta, self.model, xn, yn, dev_xn, dev_yn, self.args.lr)
            forward_elapsed = time.time() - st
                        
            log_str = "epoch {} | dev area loss {:.4f}\n".format(e, area_loss.item())
            log_str += "All Dev Losses: {}".format(all_logging_losses)
            self.print_and_save(log_str)

            # self.evaluate(e, theta, xn, yn, test_xn, test_yn)

            area_loss.backward()
            global max_grad, min_grad, max_grad_epoch, min_grad_epoch
            print_rank("max grad", max_grad, "min grad", min_grad, "max grad epoch", max_grad_epoch, "min grad epoch", min_grad_epoch)
            max_grad = 0
            min_grad = 1e6
            backward_elapsed = time.time() - st - forward_elapsed
            
            self.optimizer.step()
            self.scheduler.step()
            step_elapsed = time.time() - st - forward_elapsed - backward_elapsed
            
            log_str = "Forward Elapsed: {:.4f} | Backward Elapsed: {:.4f} | Step Elapsed: {:.4f}\n\n".format(
                forward_elapsed, backward_elapsed, step_elapsed)
            self.print_and_save(log_str)
            
            self.save(e)

    def evaluate(self, e, theta, xn, yn, test_xn, test_yn):
        with torch.no_grad():
            area_loss, all_losses, all_logging_losses = self.gamma_model(
                theta, self.model, xn, yn, test_xn, test_yn, self.args.lr, mode="test")

            log_str = "epoch {} | test area loss {:.4f}\n".format(e, area_loss.item())
            log_str += "All Test Losses: {}".format(all_logging_losses)
            self.print_and_save(log_str)
       
    def print_and_save(self, log_str):
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
         
    def save(self, epoch):
        sd = self.gamma_model.state_dict()
        gamma_t = torch.stack([sd[f"gamma.{t}"] for t in range(self.args.epochs)], dim=0)
        # print(gamma_t)
        # print(torch.sum(gamma_t, dim=1))
        save_path = os.path.join(self.args.save, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        torch.save(gamma_t, os.path.join(save_path, f"opt_gamma.pt"))
              