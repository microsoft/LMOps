# run vanilla zero-shot baseline
# LLM predictions will be in $PWD/my_data/experiment/uprise/${TASK}_${ENGINE}_ZeroShot_pred.json
python inference_openai.py \
	prompt_file=$PWD/my_data/experiment/uprise/${TASK}_0Shot_pred.json \
	output_file=$PWD/my_data/experiment/uprise/${TASK}_${ENGINE}_ZeroShot_pred.json \
	task_name=${TASK} \
	res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
	engine=${ENGINE} \
    cache_dir=${CACHE_DIR} 


# run vanilla zero-shot baseline
# LLM predictions will be in $PWD/my_data/experiment/uprise/${TASK}_${ENGINE}_UPRISE_pred.json
python inference_openai.py \
	prompt_file=$PWD/my_data/experiment/uprise/${TASK}_UPRISE_pred.json \
	output_file=$PWD/my_data/experiment/uprise/${TASK}_${ENGINE}_UPRISE_pred.json \
	task_name=${TASK} \
	res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
	engine=${ENGINE} \
    cache_dir=${CACHE_DIR} 

# the vanilla zero-shot and UPRISE evaluation results are in '$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt'

