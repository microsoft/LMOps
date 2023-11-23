#!/bin/bash

GPU=$1
DS=$2
START=$3
END=$4
for i in $(seq $START $END)
do
   CUDA_VISIBLE_DEVICES=${GPU} conda run -n cdsicd bash bin/unifiedqa_ppl_ft_train.sh ${DS}_$i
done

