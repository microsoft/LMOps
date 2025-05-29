# OPO: On-Policy RL with Optimal Reward Baseline

This repository contains the implementation of **On-Policy RL with Optimal Reward Baseline (OPO)**. Our method introduces two key components: **exact on-policy training** and **optimal reward baseline**.
The implementation is built upon the [verl](https://github.com/volcengine/verl) library, specifically using version `v0.2.0`.

## Method Overview

OPO's core enhancements are integrated with minimal modifications to existing algorithms. The implementation focuses on adjusting hyperparameters for exact on-policy training and modifying the advantage computation function to incorporate the optimal reward baseline.

### Exact On-Policy Training

To enable exact on-policy training, we set specific hyperparameters within the `verl` framework. This configuration ensures that the policy updates are strictly based on data collected from the current policy. The adjustments are:

* Setting `train_batch_size` equal to `ppo_mini_batch_size`.
* Setting both KL divergence and entropy coefficients to zero.

This can be achieved by running the following command:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${GLOBAL_BSZ} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    ...
```

### Optimal Reward Baseline

The optimal reward baseline is integrated by modifying the advantage computation function. This modification is implemented in `verl/trainer/ppo/core_algos.py` file. We have extended both the GRPO and Reinforce++ algorithms to support this optimal reward baseline.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{hao2025opo,
  title={On-Policy RL with Optimal Reward Baseline},
  author={Yaru Hao and Li Dong and Xun Wu and Shaohan Huang and Zewen Chi and Furu Wei},
  journal={arXiv preprint},
  year={2025}
}
```
