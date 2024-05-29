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

    # ================================== Scoring Stage ===================================
    prompt_pool_dir = os.path.join(args.output_dir, 'prompt_pool')
    random_sample_dir = os.path.join(args.output_dir, 'find_random') # change file name is ok
    scored_dir = os.path.join(args.output_dir, 'scored')

    exp_name = f'{args.train_clusters}'
    exp_path = os.path.join(args.output_dir, 'experiment', exp_name)
    model_weight_path = os.path.join(exp_path, args.model_folder)
    os.makedirs(exp_path, exist_ok=True)

    random_port = random.randint(21966,25000)

    if args.train_clusters is None:
        clusters = list(train_cluster_map.keys())
    else:
        clusters = args.train_clusters.split('+')
    score_cmd_list=[]
    for cluster in tqdm(clusters):
        for task in train_cluster_map[cluster]: # for Se2, there is only a task in a cluster
            echo_cmd = f'echo "scoring {task} task..."'
            task_cls = task_map.cls_dic[task]()
            learning_rate = task_cls.learning_rate
            prompt_pool_path = os.path.join(prompt_pool_dir, cluster, task+'_prompts.json')
            random_sample_path = os.path.join(random_sample_dir, cluster, task + '_random_samples')
            find_random_cmd=\
                f'python find_random_step1.py \
                output_path=$PWD/{random_sample_path + "_step1.json"} \
                task_name={task} +ds_size={args.ds_size} L={task_cls.finder_L} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'

            scored_train_path = os.path.join(scored_dir, cluster, task + '_scored_train')
            scored_valid_path = os.path.join(scored_dir, cluster, task + '_scored_valid')
            run_scorer_cmd = \
                f'accelerate launch --multi_gpu --num_processes {args.gpus} --main_process_port {random_port} \
                scorer.py \
                example_file=$PWD/{random_sample_path + "_step1.json"} \
                output_train_file=$PWD/{scored_train_path + "_step1.json"} \
                output_valid_file=$PWD/{scored_valid_path + "_step1.json"} \
                batch_size={task_cls.run_scorer_bsz} task_name={task}  model_name={args.scr_model} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'
            
            find_random_step2=\
                f'python find_random_step2.py \
                output_path=$PWD/{random_sample_path + "_step2.json"} \
                train_path=$PWD/{scored_train_path + "_step1.json"} \
                valid_path=$PWD/{scored_valid_path + "_step1.json"} \
                task_name={task} +ds_size={args.ds_size} L={task_cls.finder_L} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'
            run_scorer_step2 = \
                f'accelerate launch --multi_gpu --num_processes {args.gpus} --main_process_port {random_port} \
                scorer.py \
                example_file=$PWD/{random_sample_path + "_step2.json"} \
                output_train_file=$PWD/{scored_train_path + "_step2.json"} \
                output_valid_file=$PWD/{scored_valid_path + "_step2.json"} \
                batch_size={task_cls.run_scorer_bsz} task_name={task} model_name={args.scr_model} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'

            find_random_step3= \
                f'python find_random_step3.py \
                output_path=$PWD/{random_sample_path + "_step3.json"} \
                train_path=$PWD/{scored_train_path + "_step2.json"} \
                valid_path=$PWD/{scored_valid_path + "_step2.json"} \
                task_name={task} +ds_size={args.ds_size} L={task_cls.finder_L} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'

            run_scorer_step3 = \
                f'accelerate launch --multi_gpu --num_processes {args.gpus} --main_process_port {random_port} \
                scorer.py \
                example_file=$PWD/{random_sample_path + "_step3.json"} \
                output_train_file=$PWD/{scored_train_path + "_step3.json"} \
                output_valid_file=$PWD/{scored_valid_path + "_step3.json"} \
                batch_size={task_cls.run_scorer_bsz} task_name={task} model_name={args.scr_model} \
                prompt_pool_path=$PWD/{prompt_pool_path} \
                cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir=$PWD/{exp_path}'
            score_cmd_list += [echo_cmd, find_random_cmd, run_scorer_cmd, find_random_step2, run_scorer_step2, find_random_step3, run_scorer_step3]

    merge_data_cmd = \
        f'python merge_data.py \
        step1_train=$PWD/{scored_train_path + "_step1.json"} \
        step1_valid=$PWD/{scored_valid_path + "_step1.json"} \
        step2_train=$PWD/{scored_train_path + "_step2.json"} \
        step2_valid=$PWD/{scored_valid_path + "_step2.json"} \
        step3_train=$PWD/{scored_train_path + "_step3.json"} \
        step3_valid=$PWD/{scored_valid_path + "_step3.json"} \
        train_output_path=$PWD/{scored_train_path + ".json"} \
        valid_output_path=$PWD/{scored_valid_path + ".json"}'
    
    echo_cmd = f'echo "merging multi-step data..."'
    score_cmd_list += [echo_cmd, merge_data_cmd]
    score_cmd_list = [wrap(cmd) for cmd in score_cmd_list]
    
    # write score cmds in score.sh (default)
    with open(f"{exp_path}/{args.score_cmd_name}","w") as f:
        f.write("\n\n".join(score_cmd_list))
    print('saved scoring cmds to: ', f"{exp_path}/{args.score_cmd_name}")

    # ================================== Training Stage ===================================
    train_cmd_list = []
    echo_cmd = f'echo "start training the retriever..."'
    # print("scored_dir: ", scored_dir)
    train_retriever_cmd = \
        f'python DPR/train_dense_encoder.py \
        train_datasets=[se2_dataset] dev_datasets=[se2_valid_dataset] train=biencoder_se2 \
        output_dir=$PWD/{model_weight_path} \
        datasets.train_clusters={args.train_clusters} \
        datasets.train_file=$PWD/{scored_dir} \
        datasets.valid_file=$PWD/{scored_dir} \
        datasets.hard_neg=true datasets.top_k={args.retriever_top_k} \
        train.hard_negatives={args.retriever_top_k} \
        train.batch_size={args.retriever_bsz} \
        train.num_train_epochs={args.retriever_epoch} \
        train.learning_rate={learning_rate} \
        datasets.prompt_pool_path=$PWD/{prompt_pool_dir} \
        datasets.prompt_setup_type={args.retriever_prompt_setup} datasets.task_setup_type=q \
        encoder.cache_dir=$PWD/{args.cache_dir} \
        hydra.run.dir=$PWD/{exp_path}'

    train_cmd_list += [echo_cmd, train_retriever_cmd]

    
    train_cmd_list = [wrap(cmd) for cmd in train_cmd_list]

    # write train cmds in train.sh (defaulte)
    with open(f"{exp_path}/{args.train_cmd_name}","w") as f:
        f.write("\n\n".join(train_cmd_list))
    print('saved training cmds to: ', f"{exp_path}/{args.train_cmd_name}")

    # ================================== Inference Stage ===================================
    inference_cmd_list = []
    # 1. encode the whole prompt pool, using prompt encoder of the trained retriever
    echo_cmd = f'echo "encoding the whole prompt pool..."'
    gen_emb_cmd = \
        f"python DPR/generate_dense_embeddings.py model_file=$PWD/{model_weight_path}/dpr_biencoder.best_valid \
        ctx_src=dpr_se2 shard_id=0 num_shards=1 \
        out_file=$PWD/{model_weight_path}/dpr_enc_index \
        ctx_sources.dpr_se2.train_clusters={args.train_clusters} \
        ctx_sources.dpr_se2.prompt_pool_path=$PWD/{prompt_pool_dir} \
        ctx_sources.dpr_se2.prompt_setup_type={args.infer_prompt_setup} \
        encoder.cache_dir=$PWD/{args.cache_dir} \
        hydra.run.dir=$PWD/{exp_path}"
    
    inference_cmd_list += [echo_cmd, gen_emb_cmd]

    def get_inference_cmd(shot_num=3, retriever='se2', retrieve_prompts_outpath=""):

        assert retriever in [None, 'Random', 'Bm25', 'Sbert', 'se2']
        random = True if retriever == "random" else False

        cmd_list = []
        echo_cmd = f'echo "running inference on {task} with {retriever} retriever..."'
        cmd_list.append(echo_cmd)
        pred_outpath = os.path.join(exp_path, f'preds_for_{cluster}', f'{task}_prompts{args.shot_num}_retriever{retriever}_preds')

        if retriever == 'se2':
            pred_outpath += '_0.json'
            retrieve_prompts_outpath += "_beam_score_0.json"
        else:
            pred_outpath += '.json'
            retrieve_prompts_outpath += ".json"
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
            shot_num={shot_num} \
            random_sample={random} random_seed=42 \
            cache_dir=$PWD/{args.cache_dir} \
            hydra.run.dir=$PWD/{exp_path}"
        cmd_list.append(run_inference_cmd)
        return cmd_list

    # 2. retrieve positive prompts from the prompt pool, for each task in the testing clusters:    
    test_clusters = args.test_clusters.split('+')
    for cluster in test_clusters:
        eval_res_outpath = os.path.join(exp_path, f'eval_res_for_{cluster}.txt')
        for task in test_cluster_map[cluster]:
            echo_cmd = f'echo "se2 retrieves on {task}..."'
            retrieve_prompts_outpath = os.path.join(exp_path, f'se2_prompts_for_{cluster}', f'{task}_prompts')
            retrieve_prompts_cmd = \
                f'python DPR/dense_retriever_beam_score.py model_file=$PWD/{model_weight_path}/dpr_biencoder.best_valid \
                qa_dataset=qa_se2 ctx_datatsets=[dpr_se2] \
                encoded_ctx_files=["$PWD/{model_weight_path}/dpr_enc_index_*"]\
                out_file=$PWD/{retrieve_prompts_outpath + ".json"} \
                datasets.qa_se2.task_name={task} \
                datasets.qa_se2.task_setup_type=q  \
                datasets.qa_se2.cache_dir=$PWD/{args.cache_dir} \
                beam_size={args.beam_size} \
                shot_num={args.shot_num} \
                ctx_sources.dpr_se2.prompt_pool_path=$PWD/{prompt_pool_dir} \
                ctx_sources.dpr_se2.train_clusters={args.train_clusters} \
                ctx_sources.dpr_se2.prompt_setup_type={args.infer_prompt_setup} \
                encoder.cache_dir=$PWD/{args.cache_dir} \
                hydra.run.dir={exp_path}'
            inference_cmd_list += [echo_cmd, retrieve_prompts_cmd]

            # se2 do not need to set shot_num here.
            inference_cmd_list += get_inference_cmd(shot_num=0, retriever='se2', retrieve_prompts_outpath=retrieve_prompts_outpath)

            # Ablations: random, bm25 and sbert
            if args.retrieve_random:
                inference_cmd_list += get_inference_cmd(shot_num=args.shot_num, retriever='Random')
            if args.retrieve_bm25:
                echo_cmd = f'echo "bm25 retrieves on {task} task of {cluster} cluster..."'
                retrieve_prompts_outpath = os.path.join(exp_path, f'bm25_prompts_for_{cluster}', f'{task}_prompts.json')
                retrieve_bm25_prompts_cmd = \
                    f'python retrieve_bm25.py \
                    train_clusters={args.train_clusters} \
                    task_name={task} cache_dir=$PWD/{args.cache_dir} \
                    prompt_pool_path=$PWD/{prompt_pool_dir} \
                    out_file=$PWD/{retrieve_prompts_outpath} \
                    prompt_setup_type={args.retriever_prompt_setup} n_docs={args.shot_num} \
                    hydra.run.dir=$PWD/{exp_path} '
                inference_cmd_list += [echo_cmd, retrieve_bm25_prompts_cmd]
                inference_cmd_list += get_inference_cmd(shot_num=args.shot_num, retriever='Bm25')
            if args.retrieve_sbert:
                echo_cmd = f'echo "sbert retrieves on {task} task of {cluster} cluster..."'
                retrieve_prompts_outpath = os.path.join(exp_path, f'sbert_prompts_for_{cluster}', f'{task}_prompts.json')
                retrieve_sbert_prompts_cmd = \
                    f'python retrieve_sbert.py \
                    train_clusters={args.train_clusters} \
                    task_name={task} cache_dir=$PWD/{args.cache_dir} \
                    prompt_pool_path=$PWD/{prompt_pool_dir} \
                    out_file=$PWD/{retrieve_prompts_outpath} \
                    prompt_setup_type={args.retriever_prompt_setup} n_docs={args.shot_num} \
                    hydra.run.dir=$PWD/{exp_path} '
                inference_cmd_list += [echo_cmd, retrieve_sbert_prompts_cmd]
                inference_cmd_list += get_inference_cmd(shot_num=args.shot_num, retriever='Sbert')
                
    inference_cmd_list = [wrap(cmd) for cmd in inference_cmd_list]

    # write run.sh
    with open(f"{exp_path}/{args.infer_cmd_name}","w") as f:
        f.write("\n\n".join(inference_cmd_list))
    print('saved inference cmds to: ', f"{exp_path}/{args.infer_cmd_name}")

    return     

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', 
                        type=str, help='Directory for saving all the intermediate and final outputs.', 
                        default='my_data')
    parser.add_argument('--cache_dir', 
                        type=str, help='Directory for caching the huggingface models and datasets.', 
                        default='$PWD/cache')
    parser.add_argument('--gpus', 
                        type=int, help='number of gpus to use',
                        default=8)
    parser.add_argument('--score_cmd_name', 
                        type=str, help='write score cmd to this file',
                        default='score.sh')
    parser.add_argument('--train_cmd_name', 
                        type=str, help='write train cmd to this file',
                        default='train.sh')               
    parser.add_argument('--infer_cmd_name', 
                        type=str, help='write infer cmd to this file',
                        default='infer.sh')
    parser.add_argument('--model_folder', 
                        type=str, help='write model ckpt to this folder',
                        default='model_ckpt')
    # training
    parser.add_argument('--train_clusters', 
                        type=str, 
                        help='we use single task in our experience, so it will be a task name',
                        default="copa")
    parser.add_argument('--retriever_prompt_setup', 
                        type=str,
                        help='setup type of prompt, qa for question + answer, q for question only',
                        default="qa")
    parser.add_argument('--infer_prompt_setup', 
                        type=str,
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
                        help="we use single task in our experience")
    parser.add_argument('--retriever_top_k', 
                        type=int,
                        help='number of k (hard) negatives for training the retriever',
                        default=20)
    parser.add_argument('--retriever_bsz', 
                        type=int,
                        help='sum of batch size of all gpus, NOT per gpu',
                        default=8)
    parser.add_argument('--retriever_epoch', 
                        type=int,
                        help='maximum training epoch',
                        default=6)
    
    # inference
    parser.add_argument('--inf_model', 
                        type=str,
                        help='Huggingface model for inference',
                        default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--test_clusters', 
                        type=str, 
                        help='we use single task in our experience, so it will be a task name',
                        default="copa")
    parser.add_argument('--beam_size', 
                        type=int, 
                        help='beam size of the example to be searched during inference',
                        default=3)
    parser.add_argument('--shot_num', 
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
    print(args)
    get_cmds(args)
    