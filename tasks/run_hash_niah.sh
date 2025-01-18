#!/bin/bash

# # glm4-ruler
# RBIT=128
# RATIO=2048
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/glm-4-9b-chat-${RBIT}/"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=6 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name glm-4-9b-chat \
#     --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
#     --model_maxlen 131072 \
#     --dataset_path /nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K \
#     --dataset_name ruler \
#     --output_dir ./preds/hash-25-01-18/ruler-glm-4-9b-chat-128k-rbit${RBIT}-top${RATIO}-learnHash \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 0

# # glm4-longbench
# RBIT=128
# RATIO=384
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/glm-4-9b-chat-${RBIT}/"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=6 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name glm-4-9b-chat \
#     --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
#     --model_maxlen 130432 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-glm-4-9b-chat-128k-rbit${RBIT}-top${RATIO}-learnHash \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 0

# # glm4-longbench
# RBIT=128
# RATIO=256
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/glm-4-9b-chat-${RBIT}/"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=6 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name glm-4-9b-chat \
#     --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
#     --model_maxlen 130432 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-glm-4-9b-chat-128k-rbit${RBIT}-top${RATIO}-learnHash \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 0

# # glm4-longbench-v2
# RBIT=128
# RATIO=1024
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/glm-4-9b-chat-${RBIT}/"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=6 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name glm-4-9b-chat \
#     --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
#     --model_maxlen 130432 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench-v2 \
#     --dataset_name longbench-v2 \
#     --output_dir ./preds/hash-25-01-18/longbench-v2-glm-4-9b-chat-128k-rbit${RBIT}-top${RATIO}-learnHash_v02/ \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 0

# # llama3.1-ruler
# RBIT=128
# RATIO=2048
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/Meta-Llama-3.1-8B-Instruct-128"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=18 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name Meta-Llama-3.1-8B-Instruct \
#     --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --model_maxlen 131072 \
#     --dataset_path /nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K \
#     --dataset_name ruler \
#     --output_dir ./preds/hash-25-01-17/ruler-Meta-Llama-3.1-8B-Instruct-128k-rbit${RBIT}-top${RATIO}-learnHash_v6/ \
#     --method hash --write_in_time --mp_num 4 --pp_num 2 --e --min_seq_len 0

# # llama3.1-longbench
# RBIT=128
# RATIO=256
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/Meta-Llama-3.1-8B-Instruct-128"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=18 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name Meta-Llama-3.1-8B-Instruct \
#     --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --model_maxlen 130432 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-Meta-Llama-3.1-8B-Instruct-128k-rbit${RBIT}-top${RATIO}-learn_hash/ \
#     --method hash --write_in_time --mp_num 3 --pp_num 2 --e --min_seq_len 0

# # longchat-longbench
# RBIT=128
# RATIO=512
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/longchat-7b-v1.5-32k-128-v5"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=18 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name longchat-7b-v1.5-32k \
#     --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
#     --model_maxlen 32128 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-gt4096-longchat-7b-v1.5-32k-rbit${RBIT}-top${RATIO}-learn_hash-v5/ \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096

# # longchat-longbench
# RBIT=128
# RATIO=256
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/longchat-7b-v1.5-32k-128-v5"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=18 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred1.py \
#     --model_name longchat-7b-v1.5-32k \
#     --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
#     --model_maxlen 32128 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-gt4096-longchat-7b-v1.5-32k-rbit${RBIT}-top${RATIO}-learn_hash-v5/ \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096

# # longchat-longbench
# RBIT=128
# RATIO=512
# HASH_PATH="/root/workspace/myoffloading/model_weights_v4/Llama-2-7B-32K-Instruct-128"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_MEM=18 HASH_WEIGHTS_PATH=${HASH_PATH} TOPK_RATIO=${RATIO} RBIT=${RBIT} USE_NORM=1 NUM_SINK=64 NUM_RECENT=32 python3 run_pred.py \
#     --model_name Llama-2-7B-32K-Instruct \
#     --model_name_or_path  /nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct \
#     --model_maxlen 32128 \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench \
#     --dataset_name longbench \
#     --output_dir ./preds/hash-25-01-18/longbench-gt4096-Llama-2-7B-32K-Instruct-rbit${RBIT}-top${RATIO}-learn_hash-v1/ \
#     --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096

RCHANNEL=32
TOPK_RATIO=512
PCA_PATH=/root/workspace/myoffloading/loki_pca/Meta-Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RCHANNEL=${RCHANNEL} TOPK_RATIO=${TOPK_RATIO} NUM_SINK=64 NUM_RECENT=32 CUDA_MEM=17.0 PCA_PATH=${PCA_PATH} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_maxlen 60000 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench \
    --dataset_name longbench \
    --output_dir ./preds/loki/longbench-gt4096-Meta-Llama-3.1-8B-Instruct-128k-channel${RCHANNEL}-top${TOPK_RATIO}-sink64-recent32/ \
    --method loki --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096

RCHANNEL=32
TOPK_RATIO=512
PCA_PATH=/root/workspace/myoffloading/loki_pca/glm-4-9b-chat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RCHANNEL=${RCHANNEL} TOPK_RATIO=${TOPK_RATIO} NUM_SINK=64 NUM_RECENT=32 CUDA_MEM=17.0 PCA_PATH=${PCA_PATH} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --model_maxlen 60000 \
    --dataset_path /nfs/shared_LLM_dataset/LongBench \
    --dataset_name longbench \
    --output_dir ./preds/loki/longbench-gt4096-glm-4-9b-chat-128k-channel${RCHANNEL}-top${TOPK_RATIO}-sink64-recent32/ \
    --method loki --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096

RCHANNEL=32
TOPK_RATIO=2048
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RCHANNEL=${RCHANNEL} TOPK_RATIO=${TOPK_RATIO} PCA_PATH=${PCA_PATH} NUM_SINK=64 NUM_RECENT=32 CUDA_MEM=6.0 python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path  /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --model_maxlen 131072 \
    --dataset_path /nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K \
    --dataset_name ruler \
    --output_dir ./preds/sparq/ruler-glm-4-9b-chat-128K-channel${RCHANNEL}-top${TOPK_RATIO}-sink64-recent32/ \
    --method sparq --write_in_time --mp_num 8 --pp_num 1 --min_seq_len 0
