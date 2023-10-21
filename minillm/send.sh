for id in 1 2 3
do
ssh node-${id} "mkdir -p ~/LMOps/minillm"
rsync -avzP ./* node-${id}:~/LMOps/minillm \
    --exclude "checkpoints" \
    --exclude "downstream_data" \
    --exclude "pretrain_data" \
    --exclude "processed_data" \
    --exclude "processed_data_1" \
    --exclude "results" \
    --exclude "*__pychache__/*" \
    --exclude "*.egg-info" \
    --exclude "*.pyc" \
    --omit-dir-times
done