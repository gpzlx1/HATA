#!/bin/bash

# hash
for RBIT in 128
do
    for RATIO in {0.1,0.05,0.03}
    do
        HASH_PATH="/root/workspace/myoffloading/model_weights/longchat-7b-v1.5-32k-${RBIT}/"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} python3 run_pred.py \
            --model_name longchat-7b-v1.5-32k \
            --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
            --model_maxlen 32000 \
            --dataset_path . \
            --dataset_name niah \
            --output_dir ./preds/hash/niah-longchat-rbit${RBIT}-top${RATIO}/ \
            --method hash --write_in_time --mp_num 8 --e --min_seq_len 0
        python eval_niah.py --input preds/hash/niah-longchat-rbit${RBIT}-top${RATIO} --method hash-${RBIT}-${RATIO} --model longchat-7b-v1.5-32k
    done
done
