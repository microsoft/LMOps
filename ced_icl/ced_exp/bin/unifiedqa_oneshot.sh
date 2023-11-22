#/bin/bash 

seed=42
DATASET=$1
ONESHOTIDX=$2

cd ../../t-few
python -m src.pl_train -c t03b.json+ia3.json+unifiedqa.json+qa_1shot.json \
-k load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" \
exp_name=t03b_unifiedqa_seed${seed}_1shot_${DATASET}_${ONESHOTIDX} \
few_shot_random_seed=${seed} seed=${seed} few_shot=True \
eval_epoch_interval=2 max_valid_size=64 dev_qa_subset=$DATASET train_qa_subset=$DATASET \
allow_skip_exp=False lr=3e-2 oneshot_idx=${ONESHOTIDX}
