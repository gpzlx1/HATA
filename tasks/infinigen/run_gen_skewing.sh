python gen_skewing.py \
    --model /nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct/ \
    --max_context_length 131768 \
    --output_dir /root/workspace/myoffloading/infinigen_svd/Meta-Llama-3.1-8B-Instruct \
    --pp_num 1

python gen_skewing.py \
    --model /nfs/shared_LLM_model/THUDM/glm-4-9b-chat/ \
    --max_context_length 131768 \
    --output_dir /root/workspace/myoffloading/infinigen_svd/glm-4-9b-chat \
    --pp_num 1

python gen_skewing.py \
    --model /nfs/shared_LLM_model/togethercomputer/Llama-2-7B-32K-Instruct/ \
    --max_context_length 32768 \
    --output_dir /root/workspace/myoffloading/infinigen_svd/Llama-2-7B-32K-Instruct \
    --pp_num 1
