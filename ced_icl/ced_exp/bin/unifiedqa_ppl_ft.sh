#/bin/bash 

DS=$1
WEIGHTS="${WEIGHTS_DIR}/t03b_unifiedqa_seed42_1shot_${DS}/"
WEIGHTS="$(echo "$WEIGHTS"chkpt_*.pt)"

seed=42

cd ../
python -m src.pl_eval -c t03b.json+ia3.json+unifiedqa.json+cds_ppl.json \
-k load_weight=${WEIGHTS} \
exp_name=t03b_unifiedqa_seed${seed}_cdsppl_lg_${DS} \
few_shot_random_seed=${seed} seed=${seed} few_shot=True \
save_model=False eval_epoch_interval=1 num_shot=100 \
allow_skip_exp=False
