# ResLoRA

This is the offical implementation of ResLoRA.

## Quick Start
Our experiments run on 8 NVIDIA Tesla V100 GPU. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

### Existing Experiments
To start with existing experiments, you can run the bash scripts in `example` directory. All hyper-parameters are set in the scripts.

You should change the `res_flag`, `merge_flag` and `pre_num` in the scripts to run different models. The value of hyper-parameters means:

| parameters in paper | parameters in code |
|---------------------|--------------------|
| w/o res             | `res_flag=0`       |
| res_is              | `res_flag=1`       |
| res_bs              | `res_flag=2`       |
| res_ms              | `res_flag=3`       |
| w/o merge           | `merge_flag=0`     |
| merge_bi            | `merge_flag=3`     |
| merge_bw            | `merge_flag=4`     |


### New Experiments
It is also easy to conduct experiments in new models. To simplify the use of ResLoRA, we implement it as a wrapped model just like `peft` of Huggingface.
To use ResLoRA in a new model, you should follow the steps below:
1. Modify `TARGET_MODULES_DICT` in `reslora.py`, to add which layers in original model you want to apply ResLoRA.
2. Use `ResLoraModel` to wrap your own model, and executable `resmodel.new_epoch()` to init the ResLoRA parameters.
3. Replace `Trainer` with `ResTrainer` in your training script, which is implemented in `run_nlg.py`.
4. Train your original model just like before.

## Notes and Acknowledgments
The implementation is based on https://github.com/huggingface/transformers

We also used some code from: https://github.com/microsoft/LoRA