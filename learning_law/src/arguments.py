import argparse
import os


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument("--model-name", type=str)
    group.add_argument("--model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)

    group.add_argument("--hidden-size", type=int, default=128)
    group.add_argument("--num-head", type=int, default=8)
    group.add_argument("--num-layers", type=int, default=2)
    group.add_argument('--max-length', type=int, default=64)
    group.add_argument('--vocab-size', type=int, default=5000)
    
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument("--wandb-name", type=str, default=None)
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")

    group.add_argument("--load-cache", type=str, default=None)
    group.add_argument("--load-gamma", type=str, default=None)

    group.add_argument("--opt-gamma", action="store_true")
    group.add_argument("--toy-zero2", action="store_true")

    group.add_argument("--eval-opt-gamma", action="store_true")
    group.add_argument("--eval-no-CT", action="store_true")
    group.add_argument("--policy-name", type=str, default=None)
    group.add_argument("--eval-gamma-epochs", type=str, default=".")
    
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--test-num", type=int, default=-1)
    group.add_argument("--test-ratio", type=float, default=1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)

    group.add_argument("--train-mu", type=float, default=0)
    group.add_argument("--train-sigma", type=float, default=1)
    group.add_argument("--train-noise", type=float, default=1)
    group.add_argument("--dev-mu", type=float, default=0)
    group.add_argument("--dev-sigma", type=float, default=1)
    group.add_argument("--dev-noise", type=float, default=0.1)

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
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-gd", type=int, default=7)
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')

    group.add_argument("--outer-lr", type=float, default=0.0001)
    group.add_argument("--outer-epochs", type=int, default=5)
    group.add_argument("--opt-gamma-wm-steps", type=int, default=0)
    group.add_argument("--grad-batch-size", type=int, default=-1)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    assert args.model_type is not None

    args.n_gpu = args.n_gpu * args.n_nodes
    
    assert args.model_type is not None
        
    if args.model_type in ["transformer", "perceptron"]:
        suffix = ""
        if args.opt_gamma:
            suffix += f"opt-{args.outer_lr}-{args.opt_gamma_wm_steps}"
        if args.eval_opt_gamma:
            suffix += "eval_opt"
    
    suffix += args.save_additional_suffix
    save_path = os.path.join(
        args.save,
        args.model_type,
        args.model_name,
        (f"bs{args.batch_size}-lr{args.lr}-tn{args.train_num}-dn{args.dev_num}-e{args.epochs}"),
        suffix,
        f"{args.seed}-{args.seed_data}-{args.seed_gd}",
    )
    args.save = save_path

    return args
