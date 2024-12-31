#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_pred.py \
    --model_name Llama-3-8B-Instruct-262k \
    --model_name_or_path  /nfs/shared_LLM_model/gradientai/Llama-3-8B-Instruct-262k \
    --model_maxlen 131072 \
    --dataset_path /nfs/shared_LLM_dataset/RULER/Llama-3-8B-Instruct-262k/128K \
    --dataset_name ruler \
    --output_dir ./preds/full/ruler-Llama-3-8B-Instruct-262k/ \
    --method flashattn --write_in_time --mp_num 2 --pp_num 2 --e --min_seq_len 0
python eval_ruler.py --model preds/full/ruler-Llama-3-8B-Instruct-262k
