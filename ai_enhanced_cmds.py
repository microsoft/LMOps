import argparse
import os
import random
import textwrap
from tqdm import tqdm
from DPR.dpr.utils.tasks import task_map, train_cluster_map, test_cluster_map

def wrap(cmd):
    '''
    Wrap command for readability in shell scripts.
    '''
    bs = ' \\\n\t '
    return bs.join(textwrap.wrap(cmd, break_long_words=False, break_on_hyphens=False))

def validate_args(args):
    '''
    Validate the arguments to ensure correctness.
    '''
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} does not exist.")
    if args.gpus <= 0:
        raise ValueError("Number of GPUs must be greater than 0.")
    if args.ds_size <= 0:
        raise ValueError("Dataset size must be a positive integer.")
    if not args.train_clusters or not args.test_clusters:
        raise ValueError("Both train_clusters and test_clusters must be specified.")

def optimize_parameters():
    '''
    AI feature: Suggests optimal parameters for training based on past data.
    '''
    # Placeholder for AI logic to determine optimal parameters
    optimal_bsz = 16  # Example value
    optimal_epoch = 3  # Example value
    return optimal_bsz, optimal_epoch

def intelligent_task_selection():
    '''
    AI feature: Recommends task clusters based on the data provided.
    '''
    # Placeholder for AI logic to recommend task clusters
    recommended_clusters = "nli+common_reason"  # Example value
    return recommended_clusters

def dynamic_model_selection(task):
    '''
    AI feature: Dynamically selects the best model based on task type.
    '''
    # Placeholder for AI logic to select the best model
    if "nli" in task:
        return "EleutherAI/gpt-neo-2.7B"
    else:
        return "bert-base-uncased"

def get_cmds(args):
    '''
    Generate training and inference commands.
    '''
    validate_args(args)

    # AI-driven parameter optimization
    if args.auto_optimize:
        args.retriever_bsz, args.retriever_epoch = optimize_parameters()

    # AI-driven task selection
    if args.auto_select_tasks:
        args.train_clusters = intelligent_task_selection()

    # ================================== Train Stage ===================================
    prompt_pool_dir = os.path.join(args.output_dir, 'prompt_pool')
    random_sample_dir = os.path.join(args.output_dir, 'find_random')
    scored_dir = os.path.join(args.output_dir, 'scored')

    exp_name = f'train_{args.train_clusters}_test_{args.test_clusters}'
    exp_path = os.path.join(args.output_dir, 'experiment', exp_name)
    os.makedirs(exp_path, exist_ok=True)

    random_port = random.randint(21966, 25000)

    if args.train_clusters is None:
        clusters = list(train_cluster_map.keys())
    else:
        clusters = args.train_clusters.split('+')

    train_cmd_list = []
    for cluster in tqdm(clusters):
        for task in train_cluster_map[cluster]:
            echo_cmd = f'echo "Scoring {task} task of {cluster} cluster..."'
            task_cls = task_map.cls_dic[task]()
            prompt_pool_path = os.path.join(prompt_pool_dir, cluster, task + '_prompts.json')
            random_sample_path = os.path.join(random_sample_dir, cluster, task + '_random_samples.json')
            find_random_cmd = \
                f'python find_random.py output_path=$PWD/{random_sample_path} \
                task_name={task} +ds_size={args.ds_size} L={task_cls.finder_L} \
                prompt_pool_path=$PWD/{prompt_pool_path} cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'

            scored_train_path = os.path.join(scored_dir, cluster, task + '_scored_train.json')
            scored_valid_path = os.path.join(scored_dir, cluster, task + '_scored_valid.json')
            run_scorer_cmd = \
                f'accelerate launch --multi_gpu --num_processes {args.gpus} --main_process_port {random_port} \
                scorer.py example_file=$PWD/{random_sample_path} \
                output_train_file=$PWD/{scored_train_path} \
                output_valid_file=$PWD/{scored_valid_path} \
                batch_size={task_cls.run_scorer_bsz} task_name={task}  \
                model_name={dynamic_model_selection(task)} \
                prompt_pool_path=$PWD/{prompt_pool_path} cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'

            train_cmd_list += [echo_cmd, find_random_cmd, run_scorer_cmd]

    echo_cmd = f'echo "Start training the retriever..."'
    train_retriever_cmd = \
        f'python DPR/train_dense_encoder.py train_datasets=[uprise_dataset] dev_datasets=[uprise_valid_dataset] \
        train=biencoder_uprise output_dir=$PWD/{exp_path} \
        datasets.train_clusters={args.train_clusters} \
        datasets.train_file=$PWD/{scored_dir} \
        datasets.valid_file=$PWD/{scored_dir} \
        datasets.hard_neg=true datasets.multi_task={args.multi_task} \
        datasets.top_k={args.retriever_top_k} train.hard_negatives={args.retriever_top_k} \
        train.batch_size={args.retriever_bsz} \
        train.num_train_epochs={args.retriever_epoch} \
        datasets.prompt_pool_path=$PWD/{prompt_pool_dir} \
        datasets.prompt_setup_type={args.retriever_prompt_setup} \
        datasets.task_setup_type=q encoder.cache_dir=$PWD/{args.cache_dir} \
        hydra.run.dir=$PWD/{exp_path}'

    train_cmd_list += [echo_cmd, train_retriever_cmd]
    train_cmd_list = [wrap(cmd) for cmd in train_cmd_list]

    with open(f"{exp_path}/train.sh", "w") as f:
        f.write("\n\n".join(train_cmd_list))
    print('Saved training commands to: ', f"{exp_path}/train.sh")

    # ================================== Inference Stage ===================================
    inference_cmd_list = []
    echo_cmd = f'echo "Encoding the whole prompt pool..."'
    gen_emb_cmd = \
        f"python DPR/generate_dense_embeddings.py model_file=$PWD/{exp_path}/dpr_biencoder.best_valid \
        ctx_src=dpr_uprise shard_id=0 num_shards=1 \
        out_file=$PWD/{exp_path}/dpr_enc_index \
        ctx_sources.dpr_uprise.train_clusters={args.train_clusters} \
        ctx_sources.dpr_uprise.prompt_pool_path=$PWD/{prompt_pool_dir} \
        ctx_sources.dpr_uprise.prompt_setup_type={args.retriever_prompt_setup} \
        encoder.cache_dir=$PWD/{args.cache_dir} \
        hydra.run.dir=$PWD/{exp_path}"

    inference_cmd_list += [echo_cmd, gen_emb_cmd]

    def get_inference_cmd(num_prompts=3, retriever='uprise'):
        assert retriever in [None, 'Random', 'Bm25', 'Sbert', 'Uprise']
        random = True if retriever == "Random" else False

        echo_cmd = f'echo "Running inference on {task} task of {cluster} cluster with {retriever} retriever..."'
        pred_outpath = os.path.join(exp_path, f'preds_for_{cluster}', f'{task}_prompts{num_prompts}_retriever{retriever}_preds.json')

        run_inference_cmd = \
            f"accelerate launch --num_processes 1  --main_process_port {random_port} \
            inference.py prompt_file=$PWD/{retrieve_prompts_outpath} \
            task_name={task} \
            output_file=$PWD/{pred_outpath} \
            res_file=$PWD/{eval_res_outpath} \
            batch_size={args.inference_bsz} \
            train_clusters={args.train_clusters} \
            model_name={args.inf_model} \
            prompt_pool_path=$PWD/{prompt_pool_dir} \
            num_prompts={num_prompts} \
            random_sample={random} random_seed=42 \
            cache_dir=$PWD/{args.cache_dir} \
            hydra.run.dir=$PWD/{exp_path}"

        return [echo_cmd, run_inference_cmd]

    test_clusters = args.test_clusters.split('+')
    for cluster in test_clusters:
        eval_res_outpath = os.path.join(exp_path, f'results_for_{cluster}', 'all_in_one_res.json')
        retrieve_prompts_outpath = os.path.join(exp_path, f'results_for_{cluster}', 'all_in_one_prompts.json')
        inference_cmd_list += get_inference_cmd(num_prompts=args.num_prompts, retriever=args.retriever)

    inference_cmd_list = [wrap(cmd) for cmd in inference_cmd_list]

    with open(f"{exp_path}/inference.sh", "w") as f:
        f.write("\n\n".join(inference_cmd_list))
    print('Saved inference commands to: ', f"{exp_path}/inference.sh")

    print(f"Please run 'bash {exp_path}/train.sh' to start training the model")
    print(f"Please run 'bash {exp_path}/inference.sh' to start inference")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--ds_size', type=int, default=10)
    parser.add_argument('--retriever_top_k', type=int, default=10)
    parser.add_argument('--retriever_bsz', type=int, default=32)
    parser.add_argument('--retriever_epoch', type=int, default=5)
    parser.add_argument('--retriever_prompt_setup', type=str, default="q")
    parser.add_argument('--train_clusters', type=str, default=None)
    parser.add_argument('--test_clusters', type=str, default=None)
    parser.add_argument('--multi_task', type=bool, default=False)
    parser.add_argument('--auto_optimize', action='store_true', help="Enable AI-based parameter optimization.")
    parser.add_argument('--auto_select_tasks', action='store_true', help="Enable AI-based task selection.")
    parser.add_argument('--num_prompts', type=int, default=3)
    parser.add_argument('--retriever', type=str, default="uprise", choices=[None, "Random", "Bm25", "Sbert", "Uprise"])
    parser.add_argument('--inference_bsz', type=int, default=8)
    parser.add_argument('--inf_model', type=str, default="EleutherAI/gpt-neo-2.7B")
    
    args = parser.parse_args()
    get_cmds(args)

if __name__ == '__main__':
    main()
