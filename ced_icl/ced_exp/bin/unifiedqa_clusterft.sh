#/bin/bash 

seed=42
CLUSTER_PATH="${CLUSTER_PATH_ENV}/cluster_assignment.csv"

DATASET=$1
CLUSTER_ID="${1}_${2}"

python -m src.pl_train -c t03b.json+ia3.json+unifiedqa.json+qa_clusterft.json \
-k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" \
exp_name=t03b_unifiedqa_seed${seed}_clusterft_${CLUSTER_ID} \
few_shot_random_seed=${seed} seed=${seed} few_shot=True \
eval_epoch_interval=2 max_valid_size=64 dev_qa_subset=$DATASET \
allow_skip_exp=False lr=3e-2 $ITP cluster_ft_path=$CLUSTER_PATH cluster_id=$CLUSTER_ID
