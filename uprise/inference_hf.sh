
# retrieve prompts for each task example
# the retrieved prompts will be in '$PWD/my_data/experiment/uprise/${TASK}_prompts.json'
python DPR/dense_retriever.py \
	 model_file=${RETRIEVER} \
	 qa_dataset=qa_uprise ctx_datatsets=[dpr_uprise] \
	 encoded_ctx_files=[$PWD/my_data/experiment/uprise/dpr_enc_index_*] \
	 out_file=$PWD/my_data/experiment/uprise/${TASK}_prompts.json \
	 datasets.qa_uprise.task_name=${TASK} \
	 datasets.qa_uprise.cache_dir=${CACHE_DIR} \
	 n_docs=3 \
	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
	 encoder.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PWD/my_data/experiment/uprise

# run vanill zero-shot baseline, 
# the LLM predictions will be in '$PWD/my_data/experiment/uprise/${TASK}_0Shot_pred.json'
accelerate launch --num_processes 1 --main_process_port \
	 23548      inference.py \
	 prompt_file=$PWD/my_data/experiment/uprise/${TASK}_prompts.json \
	 task_name=${TASK} \
	 output_file=$PWD/my_data/experiment/uprise/${TASK}_0Shot_pred.json \
	 res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
	 model_name=${LLM} cache_dir=${CACHE_DIR} \
	 num_prompts=0 batch_size=8 \
	 hydra.run.dir=$PWD/my_data/experiment/uprise

# run UPRISE
# the LLM predictions will be in '$PWD/my_data/experiment/uprise/${TASK}_Uprise_pred.json'
accelerate launch --num_processes 1 --main_process_port \
	 23548       inference.py \
	 prompt_file=$PWD/my_data/experiment/uprise/${TASK}_prompts.json \
	 task_name=${TASK} \
	 output_file=$PWD/my_data/experiment/uprise/${TASK}_Uprise_pred.json \
	 res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
	 model_name=${LLM} cache_dir=${CACHE_DIR} \
	 num_prompts=3  batch_size=8 \
	 hydra.run.dir=$PWD/my_data/experiment/uprise

# the vanilla zero-shot and UPRISE evaluation results are in '$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt'