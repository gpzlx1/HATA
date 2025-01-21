#!/bin/bash

LONGBENCH_PATH=/nfs/shared_LLM_dataset/LongBench
RULER_LLAMA31_PATH=/nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K
RULER_GLM4_PATH=/nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K
NIAH_LLAMA31_PATH=/root/workspace/myoffloading/myTransformer/tasks/niah/llama31_32k_128k
NIAH_GLM4_PATH=/root/workspace/myoffloading/myTransformer/tasks/niah/glm4_32k_128k
INFINITEBENCH_PATH=/nfs/shared_LLM_dataset/xinrongzhang2022/InfiniteBench/processed
LONGBENCHV2_PATH=/nfs/shared_LLM_dataset/LongBench-v2

LLAMA31_PATH=/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
LLAMA2_PATH=/nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct
GLM4_PATH=/nfs/shared_LLM_model/THUDM/glm-4-9b-chat

LONGBENCH_BUDGET=512
RULER_BUDGET=2048
NIAH_BUDGET=2048
INFINITEBENCH_BUDGET=2048
LONGBENCHV2_BUDGET=1024

RBIT=128
NUM_SINK=0
NUM_RECENT=0
USE_NORM=1

LLAMA31_HASH_WEIGHTS_PATH=/root/workspace/myoffloading/model_weights_v5/Meta-Llama-3.1-8B-Instruct-${RBIT}
LLAMA2_HASH_WEIGHTS_PATH=/root/workspace/myoffloading/model_weights_v5/Llama-2-7B-32K-Instruct-${RBIT}
GLM4_HASH_WEIGHTS_PATH=/root/workspace/myoffloading/model_weights_v5/glm-4-9b-chat-${RBIT}

# longbenche-Full-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${LONGBENCH_BUDGET} HASH_WEIGHTS_PATH=${LLAMA31_HASH_WEIGHTS_PATH} CUDA_MEM=18.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 60000 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/hash/longbenche_gt4K_full-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/hash/longbenche_gt4K_full-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# longbenche-Full-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${LONGBENCH_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 60000 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/hash/longbenche_gt4K_full-glm-4-9b-chat-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/hash/longbenche_gt4K_full-glm-4-9b-chat-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# longbenche-32K-llama2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${LONGBENCH_BUDGET} HASH_WEIGHTS_PATH=${LLAMA2_HASH_WEIGHTS_PATH} CUDA_MEM=18.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Llama-2-7B-32K-Instruct \
    --model_name_or_path ${LLAMA2_PATH} \
    --model_maxlen 32128 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/hash/longbenche_gt4K_32K-Llama-2-7B-32K-Instruct-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/hash/longbenche_gt4K_32K-Llama-2-7B-32K-Instruct-rbit${RBIT}-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# RULER-128K-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${RULER_BUDGET} HASH_WEIGHTS_PATH=${LLAMA31_HASH_WEIGHTS_PATH} CUDA_MEM=18.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${RULER_LLAMA31_PATH} \
    --dataset_name ruler \
    --output_dir ./preds/hash/ruler_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${RULER_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 4 --pp_num 2 --min_seq_len 0
python eval_ruler.py --model preds/hash/ruler_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${RULER_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# RULER-128K-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${RULER_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${RULER_GLM4_PATH} \
    --dataset_name ruler \
    --output_dir ./preds/hash/ruler_128K-glm-4-9b-chat-rbit${RBIT}-top${RULER_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --min_seq_len 0
python eval_ruler.py --model preds/hash/ruler_128K-glm-4-9b-chat-rbit${RBIT}-top${RULER_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# NiaH-128K-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${NIAH_BUDGET} HASH_WEIGHTS_PATH=${LLAMA31_HASH_WEIGHTS_PATH} CUDA_MEM=18.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${NIAH_LLAMA31_PATH} \
    --dataset_name niah \
    --output_dir ./preds/hash/niah_32K_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${NIAH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 4 --pp_num 2 --min_seq_len 0
python eval_niah.py --input preds/hash/niah_32K_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${NIAH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM} --model Llama-3.1-8B-Instruct --method HATA

# NiaH-128K-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${NIAH_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${NIAH_GLM4_PATH} \
    --dataset_name niah \
    --output_dir ./preds/hash/niah_32K_128K-glm-4-9b-chat-rbit${RBIT}-top${NIAH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --min_seq_len 0
python eval_niah.py --input preds/hash/niah_32K_128K-glm-4-9b-chat-rbit${RBIT}-top${NIAH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM} --model GLM-4-9b-Chat --method HATA

# InfiniteBench-128K-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${INFINITEBENCH_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${INFINITEBENCH_PATH} \
    --dataset_name infinitebench \
    --output_dir ./preds/hash/infinitebench_128K-glm-4-9b-chat-rbit${RBIT}-top${INFINITEBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 8 --pp_num 1 --min_seq_len 0
python eval_longbench_infinitebench.py --model preds/hash/infinitebench_128K-glm-4-9b-chat-rbit${RBIT}-top${INFINITEBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# InfiniteBench-128K-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${INFINITEBENCH_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${INFINITEBENCH_PATH} \
    --dataset_name infinitebench \
    --output_dir ./preds/hash/infinitebench_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${INFINITEBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 4 --pp_num 2 --min_seq_len 0
python eval_longbench_infinitebench.py --model preds/hash/infinitebench_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${INFINITEBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# LongBench_v2-128K-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${LONGBENCHV2_BUDGET} HASH_WEIGHTS_PATH=${GLM4_HASH_WEIGHTS_PATH} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 130816 \
    --dataset_path ${LONGBENCHV2_PATH} \
    --dataset_name longbench-v2 \
    --output_dir ./preds/hash/longbenchv2_128K-glm-4-9b-chat-rbit${RBIT}-top${LONGBENCHV2_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 7 --pp_num 1 --min_seq_len 0
python eval_longbench_v2.py --model preds/hash/longbenchv2_128K-glm-4-9b-chat-rbit${RBIT}-top${LONGBENCHV2_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}

# LongBench_v2-128K-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RBIT=${RBIT} TOPK_RATIO=${LONGBENCHV2_BUDGET} HASH_WEIGHTS_PATH=${LLAMA31_HASH_WEIGHTS_PATH} CUDA_MEM=18.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} USE_NORM=${USE_NORM} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 130816 \
    --dataset_path ${LONGBENCHV2_PATH} \
    --dataset_name longbench-v2 \
    --output_dir ./preds/hash/longbenchv2_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${LONGBENCHV2_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}/ \
    --method hash --write_in_time --mp_num 3 --pp_num 2 --min_seq_len 0
python eval_longbench_v2.py --model preds/hash/longbenchv2_128K-Meta-Llama-3.1-8B-Instruct-rbit${RBIT}-top${LONGBENCHV2_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}-use_knorm${USE_NORM}
