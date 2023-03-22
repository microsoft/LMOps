# run vanilla zero-shot baseline
# LLM predictions will be in $PWD/experiments/${TASK}/${ENGINE}_ZeroShot_pred.json
python inference_openai.py \
	prompt_file=$PWD/experiments/${TASK}/ZeroShot_pred.json \
	output_file=$PWD/experiments/${TASK}/${ENGINE}_ZeroShot_pred.json \
	task_name=${TASK} \
	res_file=$PWD/experiments/${TASK}/evaluation_res.txt \
	engine=${ENGINE} \
    cache_dir=${CACHE_DIR} 


# run vanilla zero-shot baseline
# LLM predictions will be in $PWD/experiments/${TASK}/${ENGINE}_UPRISE_pred.json
python inference_openai.py \
	prompt_file=$PWD/experiments/${TASK}/UPRISE_pred.json \
	output_file=$PWD/experiments/${TASK}/${ENGINE}_UPRISE_pred.json \
	task_name=${TASK} \
	res_file=$PWD/experiments/${TASK}/evaluation_res.txt \
	engine=${ENGINE} \
    cache_dir=${CACHE_DIR} 

# the vanilla zero-shot and UPRISE evaluation results are in '$PWD/experiments/${TASK}/evaluation_res.txt'

