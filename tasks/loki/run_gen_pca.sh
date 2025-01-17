python gen_pca.py \
    --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset_path /nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K \
    --output_dir /root/workspace/myoffloading/loki_pca/Meta-Llama-3.1-8B-Instruct \
    --num_samples 10 --key_type pre_rotary --pp_num 4

python gen_pca.py \
    --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat \
    --dataset_path /nfs/shared_LLM_dataset/RULER/glm-4-9b-chat/128K \
    --output_dir /root/workspace/myoffloading/loki_pca/glm-4-9b-chat \
    --num_samples 1 --key_type pre_rotary --pp_num 4
