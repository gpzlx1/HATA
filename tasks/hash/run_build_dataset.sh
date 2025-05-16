#!/bin/bash

# For Meta-Llama-3.1-8B-Instruct
python build_dataset.py \
    --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --save_path /mnt/ramdisk/Meta-Llama-3.1-8B-Instruct \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template --pos_sample_ratio 0.1

# For glm-4-9b-chat
python build_dataset.py \
    --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --save_path /mnt/ramdisk/glm-4-9b-chat \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template --pos_sample_ratio 0.1

# For Llama-2-7B-32K-Instruct
python build_dataset.py \
    --model /nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct \
    --save_path /mnt/ramdisk/Llama-2-7B-32K-Instruct \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 32768 --apply_template --pos_sample_ratio 0.1

# For Qwen2.5-14B-Instruct-1M
python build_dataset.py \
    --model /nfs/shared_LLM_model/Qwen/Qwen2.5-14B-Instruct-1M \
    --save_path /mnt/ramdisk/Qwen2.5-14B-Instruct-1M \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template --pos_sample_ratio 0.1 \
    --pp_num 2

# For Qwen2.5-32B-Instruct
python build_dataset.py \
    --model /nfs/shared_LLM_model/Qwen/Qwen2.5-32B-Instruct \
    --save_path /mnt/ramdisk/Qwen2.5-32B-Instruct \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 32768 --apply_template --pos_sample_ratio 0.1 \
    --pp_num 4
