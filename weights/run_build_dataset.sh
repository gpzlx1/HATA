#!/bin/bash

python build_dataset.py \
    --model /nfs/shared_LLM_model/gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --save_path /mnt/ramdisk/Llama-3-8B-Instruct-Gradient-1048k \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template --pos_sample_ratio 0.1

python build_dataset.py \
    --model /nfs/shared_LLM_model/Qwen/Qwen2.5-7B-Instruct-1M \
    --save_path /mnt/ramdisk/Qwen2.5-7B-Instruct-1M \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template --pos_sample_ratio 0.1 \
    --pp_num 1
