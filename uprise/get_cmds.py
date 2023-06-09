'''
print out cmds for training and inference
'''

import argparse
import os
from DPR.dpr.utils.tasks import task_map, train_cluster_map, test_cluster_map
import random
import textwrap
from tqdm import tqdm


def wrap(cmd): 
    '''
    wrap cmd
    '''
    bs = ' \\\n\t '
    return bs.join(textwrap.wrap(cmd,break_long_words=False,break_on_hyphens=False))     

def get_cmds(args):

    # ================================== Train Stage ===================================
    # 1. random sample prompts and score data
    prompt_pool_dir = os.path.join(args.output_dir, 'prompt_pool')
    random_sample_dir = os.path.join(args.output_dir, 'find_random')
    scored_dir = os.path.join(args.output_dir, 'scored')

    exp_name = f'train_{args.train_clusters}_test_{args.test_clusters}'
    exp_path = os.path.join(args.output_dir, 'experiment', exp_name)
    os.makedirs(exp_path, exist_ok=True)

    random_port = random.randint(21966,25000)

    if args.train_clusters is None:
        clusters = list(train_cluster_map.keys())
    else:
        clusters = args.train_clusters.split('+')
    train_cmd_list=[]
    for cluster in tqdm(clusters):
        for task in train_cluster_map[cluster]:
            echo_cmd = f'echo "scoring {task} task of {cluster} cluster..."'
            task_cls = task_map.cls_dic[task]()
            prompt_pool_path = os.path.join(prompt_pool_dir, cluster, task+'_prompts.json')
            random_sample_path = os.path.join(random_sample_dir, cluster, task+'_random_samples.json')
            find_random_cmd=\
                f'python find_random.py output_path=$PWD/{random_sample_path} \
                task_name={task} +ds_size={args.ds_size} L={task_cls.finder_L} \
                prompt_pool_path=$PWD/{prompt_pool_path} cache_dir=$PWD/{args.cache_dir}\
                hydra.run.dir=$PWD/{exp_path}'

            scored_train_path = os.path.join(scored_dir, cluster, task+'_scored_train.json')
            scored_valid_path = os.path.join(scored_dir, cluster, task+'_scored_valid.json')
            run_scorer_cmd = \
                f'accelerate launch --multi_gpu --num_processes {args.gpus} --main_process_port {random_port} \
                scorer.py example_file=$PWD/{random_sample_path} \
                output_train_file=$PWD/{scored_train_path} \
                output_valid_file=$PWD/{scored_valid_path} \
                batch_size={task_cls.run_scorer_bsz} task_name={task}  \
                model_name={args.scr_model} \
                prompt_pool_path=$PWD/{prompt_pool_path} cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'
            
            train_cmd_list += [echo_cmd, find_random_cmd, run_scorer_cmd]

    # 2. train a retriever:
    echo_cmd = f'echo "start training the retriever..."'
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
        datasets.task_setup_type=q encoder.cache_dir=$PWD/{args.cache_dir}\
        hydra.run.dir=$PWD/{exp_path}'

    train_cmd_list += [echo_cmd, train_retriever_cmd]

    # write train cmds in train.sh
    train_cmd_list = [wrap(cmd) for cmd in train_cmd_list]

    # write run.sh
    with open(f"{exp_path}/train.sh","w") as f:
        f.write("\n\n".join(train_cmd_list))
    print('saved training cmds to: ', f"{exp_path}/train.sh")

    # ================================== Inference Stage ===================================
    inference_cmd_list = []
    # 1. encode the whole prompt pool, using prompt encoder of the trained retriever
    echo_cmd = f'echo "encoding the whole prompt pool..."'
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
        random = True if retriever == "random" else False

        echo_cmd = f'echo "running inference on {task} task of {cluster} cluster with {retriever} retriever..."'
        pred_outpath = os.path.join(exp_path, f'preds_for_{cluster}', f'{task}_prompts{args.num_prompts}_retriever{retriever}_preds.json')

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

    # 2. retrieve positive prompts from the prompt pool, for each task in the testing clusters:    
    test_clusters = args.test_clusters.split('+')
    for cluster in test_clusters:
        eval_res_outpath = os.path.join(exp_path, f'eval_res_for_{cluster}.txt')
        for task in test_cluster_map[cluster]:
            echo_cmd = f'echo "uprise retrieves on {task} task of {cluster} cluster..."'
            retrieve_prompts_outpath = os.path.join(exp_path, f'uprise_prompts_for_{cluster}', f'{task}_prompts.json')
            retrieve_prompts_cmd = \
                f'python DPR/dense_retriever.py model_file=$PWD/{exp_path}/dpr_biencoder.best_valid \
                qa_dataset=qa_uprise ctx_datatsets=[dpr_uprise] \
                encoded_ctx_files=["$PWD/{exp_path}/dpr_enc_index_*"]\
                out_file=$PWD/{retrieve_prompts_outpath} \
                datasets.qa_uprise.task_name={task} \
                datasets.qa_uprise.task_setup_type=q  \
                datasets.qa_uprise.cache_dir=$PWD/{args.cache_dir} \
                n_docs={args.num_prompts} \
                ctx_sources.dpr_uprise.prompt_pool_path=$PWD/{prompt_pool_dir} \
                ctx_sources.dpr_uprise.train_clusters={args.train_clusters} \
                ctx_sources.dpr_uprise.prompt_setup_type={args.retriever_prompt_setup} \
                encoder.cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir={exp_path}'
            inference_cmd_list += [echo_cmd, retrieve_prompts_cmd]

            # vanilla zero shot
            inference_cmd_list += get_inference_cmd(num_prompts=0, retriever=None)

            # uprise zero shot
            inference_cmd_list += get_inference_cmd(num_prompts=args.num_prompts, retriever='Uprise')

            # Ablations: replace uprise retriever with random, bm25 and sbert
            if args.retrieve_random:
                inference_cmd_list += get_inference_cmd(num_prompts=args.num_prompts, retriever='Random')
            if args.retrieve_bm25:
                echo_cmd = f'echo "bm25 retrieves on {task} task of {cluster} cluster..."'
                retrieve_prompts_outpath = os.path.join(exp_path, f'bm25_prompts_for_{cluster}', f'{task}_prompts.json')
                retrieve_bm25_prompts_cmd = \
                    f'python retrieve_bm25.py \
                    train_clusters={args.train_clusters} \
                    task_name={task} cache_dir=$PWD/{args.cache_dir} \
                    prompt_pool_path=$PWD/{prompt_pool_dir} \
                    out_file=$PWD/{retrieve_prompts_outpath} \
                    prompt_setup_type={args.retriever_prompt_setup} n_docs={args.num_prompts} \
                    hydra.run.dir=$PWD/{exp_path} '
                inference_cmd_list += [echo_cmd, retrieve_bm25_prompts_cmd]
                inference_cmd_list += get_inference_cmd(num_prompts=args.num_prompts, retriever='Bm25')
            if args.retrieve_sbert:
                echo_cmd = f'echo "sbert retrieves on {task} task of {cluster} cluster..."'
                retrieve_prompts_outpath = os.path.join(exp_path, f'sbert_prompts_for_{cluster}', f'{task}_prompts.json')
                retrieve_sbert_prompts_cmd = \
                    f'python retrieve_sbert.py \
                    train_clusters={args.train_clusters} \
                    task_name={task} cache_dir=$PWD/{args.cache_dir} \
                    prompt_pool_path=$PWD/{prompt_pool_dir} \
                    out_file=$PWD/{retrieve_prompts_outpath} \
                    prompt_setup_type={args.retriever_prompt_setup} n_docs={args.num_prompts} \
                    hydra.run.dir=$PWD/{exp_path} '
                inference_cmd_list += [echo_cmd, retrieve_sbert_prompts_cmd]
                inference_cmd_list += get_inference_cmd(num_prompts=args.num_prompts, retriever='Sbert')
                
    inference_cmd_list = [wrap(cmd) for cmd in inference_cmd_list]

    # write run.sh
    with open(f"{exp_path}/inference.sh","w") as f:
        f.write("\n\n".join(inference_cmd_list))
    print('saved inference cmds to: ', f"{exp_path}/inference.sh")

    return     

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        default='my_data')
    parser.add_argument('--cache_dir', 
                        type=str, help='Directory for caching the huggingface models and datasets.', 
                        default='../cache')
    parser.add_argument('--gpus', 
                        type=int, help='number of gpus to use',
                        default=8)
    
    # training
    parser.add_argument('--train_clusters', 
                        type=str, 
                        help='a string concatenating task clusters for training, \
                            e.g., `nli+common_reason` means nli and common_reason task clusters \
                            all supoorted clusters are in DPR.dpr.utils.tasks.train_cluster_map \
                            clusters=`all supported clsuters` when the passed value is None',
                        default=None)
    parser.add_argument('--retriever_prompt_setup', 
                        type=str,
                        help='setup type of prompt, recommend setting as `qa` for cross-task training \
                            and `q` for task-specific training',
                        default="qa")
    parser.add_argument('--ds_size', 
                        type=int,
                        help='number of maximum data examples sampled from each training dataset',
                        default=10000)
    parser.add_argument('--scr_model', 
                        type=str,
                        help='Huggingface model for scoring data',
                        default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--multi_task", 
                        action="store_true", 
                        help="True for multi-task and False for task-specific, \
                            the difference reflects on the sampling of negative prompts ONLY \
                            refer to `UpriseDataset` in `DPR/dpr/data/biencoder_data.py` for details")
    parser.add_argument('--retriever_top_k', 
                        type=int,
                        help='number of k (hard) negatives for training the retriever',
                        default=20)
    parser.add_argument('--retriever_bsz', 
                        type=int,
                        help='sum of batch size of all gpus, NOT per gpu',
                        default=16)
    parser.add_argument('--retriever_epoch', 
                        type=int,
                        help='maximum training epoch, recommend setting as `3` when cross-task training, \
                             and `10` when task-specific training',
                        default=3)
    
    # inference
    parser.add_argument('--inf_model', 
                        type=str,
                        help='Huggingface model for inference',
                        default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--test_clusters', 
                        type=str, 
                        help='a string concatenating task clusters for training, \
                            e.g., `nli+common_reason` means nli and common_reason task clusters \
                            all supoorted clusters are in DPR.dpr.utils.tasks.test_cluster_map',
                        default="nli+common_reason")
    parser.add_argument('--num_prompts', 
                        type=int, 
                        help='maximum number of retrieved prompts to be concatenated before the task input',
                        default=3)
    parser.add_argument('--retrieve_random', 
                        action="store_true", 
                        help='whether to random retrieve from our prompt pool, and run a baseline')
    parser.add_argument('--retrieve_bm25', 
                        action="store_true", 
                        help='whether to use bm25 retriever to retrieve from our prompt pool, and run a baseline')
    parser.add_argument('--retrieve_sbert', 
                        action="store_true", 
                        help='whether to use sbert to retrieve from our prompt pool, and run a baseline')
    parser.add_argument('--inference_bsz', 
                        type=int,
                        help='sum of batch size of all gpus, NOT per gpu',
                        default=1)

    args = parser.parse_args()

    get_cmds(args)
    