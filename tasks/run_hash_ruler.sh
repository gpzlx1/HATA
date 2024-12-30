#!/bin/bash

# hash
for RBIT in 128
do
    for RATIO in 0.03
    do
        HASH_PATH="/root/workspace/myoffloading/model_weights/longchat-7b-v1.5-32k-${RBIT}/"
        CUDA_VISIBLE_DEVICES=3 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=0,1,2,3 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
            --model_name longchat-7b-v1.5-32k \
            --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
            --model_maxlen 32768 \
            --dataset_path /nfs/shared_LLM_dataset/RULER/longchat-7b-v1.5-32k/32K \
            --dataset_name ruler \
            --output_dir ./preds/hash/ruler-longchat-rbit${RBIT}-top${RATIO}/ \
            --method hash --write_in_time --mp_num 4 --e --min_seq_len 0
        python eval_ruler.py --model preds/hash/ruler-longchat-rbit${RBIT}-top${RATIO}
    done
done
