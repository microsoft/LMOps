# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
import numpy as np


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default="gpt2")
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-ckpt-name", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--fp32", action="store_true")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--lm-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-rw", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    group.add_argument("--only-prompt", action="store_true")
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=-1,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--training-epochs', type=int, default=10000)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument("--kd-ratio", type=float, default=None)

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")

    return parser


def add_ppo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('ppo', 'ppo configurations')
    
    group.add_argument("--reward-scaling", type=float, default=None)
    group.add_argument("--cliprange-reward", type=float, default=1)
    group.add_argument("--ppo-epochs", type=int, default=None)
    group.add_argument("--num-rollouts", type=int, default=256)
    group.add_argument("--num-rollouts-per-device", type=int, default=None)
    group.add_argument("--cliprange", type=float, default=0.2)
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--gamma", type=float, default=0.95)
    
    return parser


def add_minillm_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('minillm', 'minillm configurations')
    
    group.add_argument("--length-norm", action="store_true")
    group.add_argument("--single-step-reg", action="store_true")
    group.add_argument("--teacher-mixed-alpha", type=float, default=None)
    group.add_argument("--lm-coef", type=float, default=1)
    
    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    
    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--peft", type=str, default=None)
    group.add_argument("--peft-lora-r", type=int, default=8)
    group.add_argument("--peft-lora-alpha", type=int, default=32)
    group.add_argument("--peft-lora-dropout", type=float, default=0.1)
    group.add_argument("--peft-name", type=str, default=None)
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--teacher-peft-name", type=str, default=None)
    group.add_argument("--teacher-peft-path", type=str, default=None)    
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_ppo_args(parser)
    parser = add_minillm_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
        
    if args.type == "eval_main":
        ckpt_name = None
        if args.ckpt_name is not None:
            ckpt_name = args.ckpt_name
        if args.peft_name is not None:
            ckpt_name = args.peft_name

        if ckpt_name is not None:
            tmp = ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)

        save_path = os.path.join(
            args.save,
            f"{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            ckpt_name,
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type == "lm":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "kd":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-kd{args.kd_ratio}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "gen":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}"),
            (f"t{args.temperature}-l{args.max_length}"),
        )
        args.save = save_path
    elif args.type == "minillm":
        ppo_prefix = f"pe{args.ppo_epochs}" + \
                     (f"_rs{args.reward_scaling}" if args.ppo_epochs is not None else "") + \
                     (f"_nr{args.num_rollouts}" if args.num_rollouts is not None else "") + \
                     (f"_ln" if args.length_norm else "") + \
                     (f"_sr" if args.single_step_reg else "") + \
                     (f"_tm{args.teacher_mixed_alpha}" if args.teacher_mixed_alpha is not None else "")
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-lm{args.lm_coef}-len{args.max_length}" + \
                (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")) + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else ""),
            ppo_prefix + args.save_additional_suffix
        )
        args.save = save_path
        args.num_rollouts_per_device = args.num_rollouts // args.n_gpu
        
        if args.warmup_iters > 0:
            assert args.scheduler_name is not None

    return args
