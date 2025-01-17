# CUDA_VISIBLE_DEVICES=0 python train_hash_weights.py \
#     --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench-v2/top9-context-only.jsonl \
#     --num_samples 9 --max_context_length 131072 --cuda_mem 0.0 --rbit 128 \
#     --save_path /root/workspace/myoffloading/model_weights_v3/glm4-128-longbenchv2-v5_multi \
#     --train_batch_size 200 --train_epochs 1 --train_iters 500 --rep_iters 100 \
#     --lr 3 --sch_iters 10 --apply_template --pp_num 1

# CUDA_VISIBLE_DEVICES=7 python train_hash_weights.py \
#     --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --dataset_path /nfs/shared_LLM_dataset/LongBench-v2/top9-context-only.jsonl \
#     --num_samples 9 --max_context_length 131072 --cuda_mem 0.0 --rbit 128 \
#     --save_path /root/workspace/myoffloading/model_weights_v3/llama3.1-128-longbenchv2-v13 \
#     --train_batch_size 1000 --train_epochs 1 --train_iters 50 --rep_iters 10 \
#     --lr 10 --sch_iters 10 --pp_num 1

CUDA_VISIBLE_DEVICES=0 python train_hash_weights.py \
    --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset_path /nfs/shared_LLM_dataset/LongBench-v2/top9-context-only.jsonl \
    --num_samples 10 --max_context_length 32768 --cuda_mem 0.0 --rbit 128 \
    --save_path /root/workspace/myoffloading/model_weights_v3/llama3.1-128-longbenchv2-v23 \
    --train_batch_size 32768 --train_epochs 10 --train_iters 1 --rep_iters 2 \
    --lr 0.1 --sch_iters 10 --pp_num 1
