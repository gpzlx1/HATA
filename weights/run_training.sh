#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights.py \
    --dataset_path /mnt/ramdisk/Llama-3-8B-Instruct-Gradient-1048k \
    --save_path ./Llama-3-8B-Instruct-Gradient-1048k-256 \
    --num_layers 32 --num_skip_layers 0 --num_heads 32 --num_kv_heads 8 \
    --head_dim 128 --rbit 256 --chunk_num 3 --mp_num 8 \
    --train_epochs 15 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 2.0 --sigma 0.1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python learn_hash_weights.py \
    --dataset_path /mnt/ramdisk/Qwen2.5-7B-Instruct-1M \
    --save_path ./Qwen2.5-7B-Instruct-1M-256 \
    --num_layers 28 --num_skip_layers 0 --num_heads 28 --num_kv_heads 4 \
    --head_dim 128 --rbit 256 --chunk_num 3 --mp_num 8 \
    --train_epochs 15 --train_iters 20 --rep_iters 10 --lr 0.1 \
    --epsilon 0.01 --lambdda 1.0 --eta 2.0 --sigma 0.1
