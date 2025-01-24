import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import myTransformer
import os
from transformers.generation.configuration_utils import GenerationConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bench_prefill_decode_speed(model, tokenizer, prefill_len, batch_size):

    warmup = 3
    prefill_bench = 5
    decode_bench = 5
    decode_step = prefill_len // 10
    input_text = "A " * prefill_len

    generation_kwargs = {
        "max_gpu_cache_memory":
        20 * 1024 * 1024 * 1024,
        "num_channels": 32,
        "sparse_ratio": 1024,
        "aux_data_path": "/root/data/loki_pca/Meta-Llama-3.1-8B-Instruct/",
        # "aux_data_path": "/root/data/loki_pca/Llama-2-7B-32K-Instruct/",
        "num_sink": 0,
        "num_recent": 0,
    }
    generation_config = GenerationConfig(**generation_kwargs)

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
    device = "cuda:6"
    torch.cuda.set_device(device)

    # model_path = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
    # model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    # model_path = "/root/data/togethercomputer/LLaMA-2-7B-32K/"
    model_path = "/root/data/meta-llama/Meta-Llama-3.1-8B-Instruct/"
    # model_path = "/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/nfs/shared_LLM_model/facebook/opt-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = torch.float16

    print(config)
    from myTransformer.models.llama.modeling_llama_loki import CustomLlamaForCausalLM
    model = CustomLlamaForCausalLM.from_pretrained(model_path,
                                                   torch_dtype=torch.float16,
                                                   config=config)
    # model.generation_config.cache_implementation = "static"
    model = model.eval().to(device)

    batch_size = 1
    print(f"batch_size: {batch_size}")
    for prefill_len in [72000]:
        bench_prefill_decode_speed(model, tokenizer, prefill_len, batch_size)
