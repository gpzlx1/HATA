#!/bin/bash

# For Meta-Llama-3.1-8B-Instruct-128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights_single_head.py \
    --dataset_path /mnt/ramdisk/Meta-Llama-3.1-8B-Instruct \
    --save_path /root/workspace/myoffloading/model_weights_v4/Meta-Llama-3.1-8B-Instruct-128 \
    --num_layers 32 --num_skip_layers 2 --num_heads 32 --num_kv_heads 8 \
    --head_dim 128 --rbit 128 --chunk_num 3 --mp_num 8 \
    --train_epochs 20 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 1.0 --sigma 0.1

# For glm-4-9b-chat
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights_single_head.py \
    --dataset_path /mnt/ramdisk/glm-4-9b-chat \
    --save_path /root/workspace/myoffloading/model_weights_v4/glm-4-9b-chat-128 \
    --num_layers 40 --num_skip_layers 2 --num_heads 32 --num_kv_heads 2 \
    --head_dim 128 --rbit 128 --chunk_num 3 --mp_num 8 \
    --train_epochs 20 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 2.0 --sigma 0.1

# For Llama-2-7B-32K-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights_single_head.py \
    --dataset_path /mnt/ramdisk/Llama-2-7B-32K-Instruct \
    --save_path /root/workspace/myoffloading/model_weights_v4/Llama-2-7B-32K-Instruct-128 \
    --num_layers 32 --num_skip_layers 2 --num_heads 32 --num_kv_heads 32 \
    --head_dim 128 --rbit 128 --chunk_num 2 --mp_num 8 \
    --train_epochs 15 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 2.0 --sigma 0.7

# For longchat-7b-v1.5-32k
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights_single_head.py \
    --dataset_path /mnt/ramdisk/longchat-7b-v1.5-32k \
    --save_path /root/workspace/myoffloading/model_weights_v4/longchat-7b-v1.5-32k-128 \
    --num_layers 32 --num_skip_layers 2 --num_heads 32 --num_kv_heads 32 \
    --head_dim 128 --rbit 128 --chunk_num 2 --mp_num 8 \
    --train_epochs 15 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 2.0 --sigma 0.7
