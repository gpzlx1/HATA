#!/bin/bash

# # For Meta-Llama-3.1-8B-Instruct
# python build_dataset.py \
#     --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --save_path /mnt/ramdisk/Meta-Llama-3.1-8B-Instruct \
#     --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
#     --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
#     --max_context_length 131072 --apply_template

# For glm-4-9b-chat
python build_dataset.py \
    --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --save_path /mnt/ramdisk/glm-4-9b-chat \
    --longbench_path /nfs/shared_LLM_dataset/LongBench/data \
    --longbench_v2_path /nfs/shared_LLM_dataset/LongBench-v2 \
    --max_context_length 131072 --apply_template
