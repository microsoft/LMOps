DOMAIN=$1
MODEL=$2
ADD_BOS=$3
MODEL_PARALLEL=$4
N_GPU=$5
OUTPUT_DIR='/tmp/output/' # for saving the prediction files
RES_DIR='/tmp/res/' # for saving the evaluation scores of each task
CACHE_DIR='/tmp/cache' # for caching hf models and datasets

if [ ${DOMAIN} == 'biomedicine' ]; then
    TASK='MQP+PubMedQA+RCT+USMLE+ChemProt'
elif [ ${DOMAIN} == 'finance' ]; then
    TASK='NER+FPB+FiQA_SA+Headline+ConvFinQA'
elif [ ${DOMAIN} == 'law' ]; then
    TASK='CaseHOLD+SCOTUS+UNFAIR_ToS'
else
    TASK=${DOMAIN}
fi

echo "Domain-specific tasks: ${TASK}"
echo "MODEL: ${MODEL}"
echo "ADD_BOS: ${ADD_BOS}"
echo "MODEL_PARALLEL: ${MODEL_PARALLEL}"
echo "N_GPU: ${N_GPU}"

if [ ${N_GPU} == '8' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${model_parallel} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '4' ]; then
    CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${model_parallel} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '2' ]; then
    CUDA_VISIBLE_DEVICES='0,1' accelerate launch  --num_processes ${N_GPU} --multi_gpu \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${model_parallel} \
        hydra.run.dir=/tmp
elif [ ${N_GPU} == '1' ]; then
    CUDA_VISIBLE_DEVICES='0' accelerate launch  --num_processes 1 \
        inference.py task_name=${TASK} model_name=${MODEL} add_bos_token=${ADD_BOS} \
        output_dir=${OUTPUT_DIR} res_dir=${RES_DIR} cache_dir=${CACHE_DIR} model_parallel=${model_parallel} \
        hydra.run.dir=/tmp
fi