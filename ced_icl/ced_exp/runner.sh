#!/bin/bash

# All commands to run experiments. Assumes 1 GPUs on 1 node. 
# Supporting more nodes / different number of GPUs can be done by changing params.
# Create conda env named cdsicd

conda run -n cdsicd pip install -r requirements.txt

# Train 1 shot in-domain models
bash bin/run_unified_oneshot.sh 0 squad2 0 31
bash bin/run_unified_oneshot.sh 0 boolq 0 31
bash bin/run_unified_oneshot.sh 0 narrativeqa 0 31
bash bin/run_unified_oneshot.sh 0 naturalqa 0 31
bash bin/run_unified_oneshot.sh 0 newsqa 0 31
bash bin/run_unified_oneshot.sh 0 npboolq 0 31
bash bin/run_unified_oneshot.sh 0 openbookqa 0 31
bash bin/run_unified_oneshot.sh 0 race 0 31


# For each model, compute scores
bash bin/run_unified_cds.sh 0 squad2 0 31
bash bin/run_unified_cds.sh 0 boolq 0 31
bash bin/run_unified_cds.sh 0 narrativeqa 0 31
bash bin/run_unified_cds.sh 0 naturalqa 0 31
bash bin/run_unified_cds.sh 0 newsqa 0 31
bash bin/run_unified_cds.sh 0 npboolq 0 31
bash bin/run_unified_cds.sh 0 openbookqa 0 31
bash bin/run_unified_cds.sh 0 race 0 31

# For training a PEFT model with in-domain examples, evaluate CED scores on training data instead of validation
bash bin/run_unified_cds_train.sh 0 squad2 0 31
bash bin/run_unified_cds_train.sh 0 boolq 0 31
bash bin/run_unified_cds_train.sh 0 narrativeqa 0 31
bash bin/run_unified_cds_train.sh 0 naturalqa 0 31
bash bin/run_unified_cds_train.sh 0 newsqa 0 31
bash bin/run_unified_cds_train.sh 0 npboolq 0 31
bash bin/run_unified_cds_train.sh 0 openbookqa 0 31
bash bin/run_unified_cds_train.sh 0 race 0 31

# For building clustered CED ICD scorers
bash bin/run_unified_clusterft.sh 0 squad2 0 31
bash bin/run_unified_clusterft.sh 0 boolq 0 31
bash bin/run_unified_clusterft.sh 0 narrativeqa 0 31
bash bin/run_unified_clusterft.sh 0 naturalqa 0 31
bash bin/run_unified_clusterft.sh 0 newsqa 0 31
bash bin/run_unified_clusterft.sh 0 npboolq 0 31
bash bin/run_unified_clusterft.sh 0 openbookqa 0 31
bash bin/run_unified_clusterft.sh 0 race 0 31


# For each model, compute scores
bash bin/run_unified_cds_cluster.sh 0 squad2 0 31
bash bin/run_unified_cds_cluster.sh 0 boolq 0 31
bash bin/run_unified_cds_cluster.sh 0 narrativeqa 0 31
bash bin/run_unified_cds_cluster.sh 0 naturalqa 0 31
bash bin/run_unified_cds_cluster.sh 0 newsqa 0 31
bash bin/run_unified_cds_cluster.sh 0 npboolq 0 31
bash bin/run_unified_cds_cluster.sh 0 openbookqa 0 31
bash bin/run_unified_cds_cluster.sh 0 race 0 31