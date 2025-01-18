python gen_pca.py \
    --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset_path /nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K \
    --output_dir /root/workspace/myoffloading/loki_pca/Meta-Llama-3.1-8B-Instruct \
    --num_samples 10 --key_type pre_rotary --pp_num 4
    --max_context_length 131072

python gen_pca.py \
    --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --dataset_path /nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K \
    --output_dir /root/workspace/myoffloading/loki_pca/glm-4-9b-chat \
    --num_samples 1 --key_type pre_rotary --pp_num 4
    --max_context_length 131072

python gen_pca.py \
    --model /nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct \
    --dataset_path /nfs/shared_LLM_dataset/RULER/Llama-2-7B-32K-Instruct/32K \
    --output_dir /root/workspace/myoffloading/loki_pca/Llama-2-7B-32K-Instruct \
    --num_samples 10 --key_type pre_rotary --pp_num 1 \
    --max_context_length 32768
