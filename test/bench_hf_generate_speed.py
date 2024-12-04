import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import myTransformer
import os
from transformers.generation.configuration_utils import GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bench_prefill_decode_speed(model, tokenizer, prefill_len, batch_size):

    warmup = 5
    prefill_bench = 10
    decode_bench = 10
    decode_step = 20
    input_text = "A " * prefill_len

    generation_kwargs = {
        "page_num": 3000,
        "page_size": 16,
        "max_gpu_cache_memory": 20212254720,  # 30GB
        "hash_rbits": 128,
        "hash_weights_path":
        "/root/workspace/myoffloading/KVOffloading/model_weights/longchat-7b-v1.5-32k-128",
        "sparse_ratio": 0.1,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    # prompt = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": f"{input_text}"}],
    #     add_generation_prompt=False,
    #     tokenize=True,
    #     return_tensors="pt",
    # )

    prompt = tokenizer.encode(input_text, return_tensors="pt")
    prompt = prompt.to(model.device)
    prompt = prompt.repeat(batch_size, 1)

    # warmup
    for i in range(warmup):
        model.generate(prompt,
                       max_new_tokens=1,
                       min_new_tokens=1,
                       generation_config=generation_config)

    # bench prefill
    torch.cuda.synchronize()
    prefill_time_begin = time.time()
    for i in range(prefill_bench):
        model.generate(prompt,
                       max_new_tokens=1,
                       min_new_tokens=1,
                       generation_config=generation_config)
    torch.cuda.synchronize()
    prefill_time_cost = (time.time() -
                         prefill_time_begin) * 1000 / prefill_bench
    print(f"prefill time cost [{prompt.shape}]: {prefill_time_cost} ms")

    # warmup
    for i in range(warmup):
        model.generate(prompt,
                       max_new_tokens=decode_step,
                       min_new_tokens=decode_step,
                       generation_config=generation_config)

    # bench decode
    torch.cuda.synchronize()
    decode_time_start = time.time()
    for i in range(decode_bench):
        model.generate(prompt,
                       max_new_tokens=decode_step,
                       min_new_tokens=decode_step,
                       generation_config=generation_config)
    torch.cuda.synchronize()
    decode_time_cost = (time.time() - decode_time_start) * 1000 / decode_bench
    decode_time_cost = decode_time_cost - prefill_time_cost
    print(
        f"decode time cost [{prompt.shape}, {decode_step}]: {decode_time_cost / (decode_step - 1)} ms"
    )


if __name__ == "__main__":
    device = "cuda:5"
    torch.cuda.set_device(device)

    # model_path = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
    model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    # model_path = "/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/nfs/shared_LLM_model/facebook/opt-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "flash_attention_2"

    print(config)
    from myTransformer.models.modeling_llama_hash import CustomLlamaForCausalLM
    model = CustomLlamaForCausalLM.from_pretrained(model_path,
                                                   torch_dtype=torch.float16,
                                                   config=config)
    model.generation_config.cache_implementation = "static"
    model = model.eval().to(device)

    batch_size = 1
    print(f"batch_size: {batch_size}")
    for prefill_len in [1000, 2000, 4000, 8000, 16000, 32000]:
        bench_prefill_decode_speed(model, tokenizer, prefill_len, batch_size)
