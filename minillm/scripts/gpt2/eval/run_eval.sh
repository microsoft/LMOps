base_path=${1-"/home/MiniLLM"}
port=2040


for data in dolly self_inst vicuna sinst uinst 
do
    # Evaluate SFT
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2-base/sft"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate KD
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2-base/kd"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate SeqKD
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2-base/seqkd"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate MiniLLM
    for seed in 10 20 30 40 50
    do
        ckpt="gpt2-base/minillm"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done
done