
# retrieve prompts for each task example
# the retrieved prompts will be in '$PWD/experiments/${TASK}/prompts.json'
python DPR/dense_retriever.py \
	 model_file=${RETRIEVER} \
	 qa_dataset=qa_epr ctx_datatsets=[dpr_epr] \
	 encoded_ctx_files=[$PWD/experiments/dpr_enc_index_*] \
	 out_file=$PWD/experiments/${TASK}/prompts.json \
	 datasets.qa_epr.task_name=${TASK} \
	 n_docs=10 \
	 ctx_sources.dpr_epr.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_epr.cache_dir=${CACHE_DIR}  \
	 datasets.qa_epr.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PWD/experiments/${TASK}

# run vanill zero-shot baseline, 
# the LLM predictions will be in '$PWD/experiments/${TASK}/ZeroShot_pred.json'
accelerate launch --num_processes 1 --main_process_port \
	 23548      inference_hf.py \
	 prompt_file=$PWD/experiments/${TASK}/prompts.json \
	 task_name=${TASK} \
	 output_file=$PWD/experiments/${TASK}/ZeroShot_pred.json \
	 res_file=$PWD/experiments/${TASK}/evaluation_res.txt \
	 model_name=${LLM} cache_dir=${CACHE_DIR} \
	 num_prompts=0 batch_size=8 \
	 hydra.run.dir=$PWD/experiments/${TASK}

# run UPRISE
# the LLM predictions will be in '$PWD/experiments/${TASK}/UPRISE_pred.json'
accelerate launch --num_processes 1 --main_process_port \
	 23548       inference_hf.py \
	 prompt_file=$PWD/experiments/${TASK}/prompts.json \
	 task_name=${TASK} \
	 output_file=$PWD/experiments/${TASK}/UPRISE_pred.json \
	 res_file=$PWD/experiments/${TASK}/evaluation_res.txt \
	 model_name=${LLM} cache_dir=${CACHE_DIR} \
	 num_prompts=3  batch_size=8 \
	 hydra.run.dir=$PWD/experiments/${TASK}

# the vanilla zero-shot and UPRISE evaluation results are in '$PWD/experiments/${TASK}/evaluation_res.txt'