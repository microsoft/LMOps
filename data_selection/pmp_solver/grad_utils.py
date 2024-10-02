from torch.func import grad, jvp, vmap, grad_and_value
from .model_wrapper import TransformerWrapper


def jvp_single(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))
    return ct


def jvp_batch(model: TransformerWrapper, batch, lam_param, params, buffers, chunk_size=None):
    return vmap(jvp_single, in_dims=(0, 0, 0, 0, None, None, None, None), chunk_size=chunk_size)(
        batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers)


def hvp_fwdrev(model: TransformerWrapper, batch, lam_param, params, buffers, bs, gacc, ws):
    f = model.compute_loss_func
    def grad_wrapper(pr):
        g = {n: 0 for n in params}
        for i in range(gacc):
            mini_batch = {k: v[i*bs:(i+1)*bs] for k, v in batch.items()}
            _g = grad(f)(pr, buffers, model, **mini_batch)
            for n in g:
                g[n] += _g[n]
        return g
    _, hvp_res = jvp(grad_wrapper, (params,), (lam_param,))
    hvp_res = model.params_to_vector(hvp_res) / (ws * gacc)
    return hvp_res
