#!/bin/bash

LONGBENCH_PATH=/nfs/shared_LLM_dataset/LongBench
RULER_LLAMA31_PATH=/nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K
RULER_GLM4_PATH=/nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K

LLAMA31_PATH=/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
LLAMA2_PATH=/nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct
GLM4_PATH=/nfs/shared_LLM_model/THUDM/glm-4-9b-chat

LONGBENCH_BUDGET=512
RULER_BUDGET=2048

NUM_SINK=64
NUM_RECENT=96

# longbenche-Full-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOPK_RATIO=${LONGBENCH_BUDGET} CUDA_MEM=17.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 60000 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/topk/longbenche_gt4K_full-Meta-Llama-3.1-8B-Instruct-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}/ \
    --method topk --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/topk/longbenche_gt4K_full-Meta-Llama-3.1-8B-Instruct-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}

# longbenche-Full-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOPK_RATIO=${LONGBENCH_BUDGET} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 60000 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/topk/longbenche_gt4K_full-glm-4-9b-chat-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}/ \
    --method topk --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/topk/longbenche_gt4K_full-glm-4-9b-chat-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}

# longbenche-32K-llama2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOPK_RATIO=${LONGBENCH_BUDGET} CUDA_MEM=17.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} python3 run_pred.py \
    --model_name Llama-2-7B-32K-Instruct \
    --model_name_or_path ${LLAMA2_PATH} \
    --model_maxlen 32128 \
    --dataset_path ${LONGBENCH_PATH} \
    --dataset_name longbench \
    --output_dir ./preds/topk/longbenche_gt4K_32K-Llama-2-7B-32K-Instruct-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}/ \
    --method topk --write_in_time --mp_num 8 --pp_num 1 --e --min_seq_len 4096
python eval_longbench_infinitebench.py --model preds/topk/longbenche_gt4K_32K-Llama-2-7B-32K-Instruct-top${LONGBENCH_BUDGET}-sink${NUM_SINK}-recent${NUM_RECENT}

# RULER-128K-llama3.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOPK_RATIO=${RULER_BUDGET} CUDA_MEM=17.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} python3 run_pred.py \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --model_name_or_path ${LLAMA31_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${RULER_LLAMA31_PATH} \
    --dataset_name ruler \
    --output_dir ./preds/topk/ruler_128K-Meta-Llama-3.1-8B-Instruct-top${TOPK_RATIO}-sink${NUM_SINK}-recent${NUM_RECENT}/ \
    --method topk --write_in_time --mp_num 4 --pp_num 2 --min_seq_len 0
python eval_ruler.py --model preds/topk/ruler_128K-Meta-Llama-3.1-8B-Instruct-top${TOPK_RATIO}-sink${NUM_SINK}-recent${NUM_RECENT}

# RULER-128K-glm4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOPK_RATIO=${RULER_BUDGET} CUDA_MEM=6.0 NUM_RECENT=${NUM_RECENT} NUM_SINK=${NUM_SINK} python3 run_pred.py \
    --model_name glm-4-9b-chat \
    --model_name_or_path ${GLM4_PATH} \
    --model_maxlen 131072 \
    --dataset_path ${RULER_GLM4_PATH} \
    --dataset_name ruler \
    --output_dir ./preds/topk/ruler_128K-glm-4-9b-chat-top${TOPK_RATIO}-sink${NUM_SINK}-recent${NUM_RECENT}/ \
    --method topk --write_in_time --mp_num 8 --pp_num 1 --min_seq_len 0
python eval_ruler.py --model preds/topk/ruler_128K-glm-4-9b-chat-top${TOPK_RATIO}-sink${NUM_SINK}-recent${NUM_RECENT}
