#!/bin/bash

# longchat-7b-v1.5-32k
for RATIO in 0.03
do
    SKEWING_PATH="/root/workspace/myoffloading/infinigen_skewing/longchat-7b-v1.5-32k_skewing"
    CUDA_VISIBLE_DEVICES=0,1,2,3 SKEWING_PATH=${SKEWING_PATH} TOPK_RATIO=${RATIO} python3 run_pred.py \
        --model_name longchat-7b-v1.5-32k \
        --model_name_or_path  /nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k \
        --model_maxlen 31500 \
        --dataset_path /nfs/shared_LLM_dataset/LongBench \
        --dataset_name longbench \
        --output_dir ./preds/infinigen/longbench-longchat-7b-v1.5-32k_skewing-38channels-top${RATIO}/ \
        --method infinigen --write_in_time --mp_num 4 --e --min_seq_len 0
    python eval_longbench_infinitebench.py --model preds/infinigen/longbench-longchat-7b-v1.5-32k_skewing-38channels-top${RATIO}
done

# # Meta-Llama-3.1-8B-Instruct
# for RATIO in {0.03,0.05,0.1}
# do
#     SKEWING_PATH="/root/workspace/myoffloading/infinigen_skewing/Meta-Llama-3.1-8B-Instruct_skewing"
#     CUDA_VISIBLE_DEVICES=0,1,2,3 SKEWING_PATH=${SKEWING_PATH} TOPK_RATIO=${RATIO} python3 run_pred.py \
#         --model_name Meta-Llama-3.1-8B-Instruct \
#         --model_name_or_path  /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --model_maxlen 31500 \
#         --dataset_path /nfs/shared_LLM_dataset/LongBench \
#         --dataset_name longbench \
#         --output_dir ./preds/infinigen/longbench-Meta-Llama-3.1-8B-Instruct-38channels-top${RATIO}/ \
#         --method infinigen --write_in_time --mp_num 4 --e --min_seq_len 0
#     python eval_longbench_infinitebench.py --model preds/infinigen/longbench-Meta-Llama-3.1-8B-Instruct-38channels-top${RATIO}
# done
