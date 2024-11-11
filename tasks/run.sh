#!/bin/bash

python3 run_pred.py \
    --model_name longchat-7b-v1.5-32k \
    --model_name_or_path /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
    --model_maxlen 31500 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench \
    --output_dir ./pred_test \
    --method flashattn --mp_num 8 --min_seq_len 0 --e