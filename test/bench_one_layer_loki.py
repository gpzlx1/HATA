import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import os
from myTransformer.models.llama.modeling_llama_loki import CustomLlamaDecoderLayer
from myTransformer.cache.kvcache_loki import LokiStaticCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bench(layer: CustomLlamaDecoderLayer,
          kvcache: LokiStaticCache,
          batch_size,
          prefill_len,
          decode_step,
          device,
          dtype=torch.float16):

    warmup = 3
    bench_epoch = 5
    hidden_size = layer.self_attn.hidden_size

    prefill_position_ids = torch.arange(0, prefill_len,
                                        device=device).unsqueeze(0)

    prefill_time = []
    decode_time = []

    with torch.no_grad():

        # warmup
        for i in range(warmup):
            kvcache.reset(batch_size)

            prefill_hidden = torch.randn(
                (batch_size, prefill_len, hidden_size),
                dtype=dtype,
                device=device).view(-1, hidden_size)

            decode_hidden = [
                torch.randn((batch_size, 1, hidden_size),
                            dtype=dtype,
                            device=device).view(-1, hidden_size)
                for _ in range(decode_step)
            ]

            kvcache.alloc(prefill_len)
            layer(prefill_hidden, None, prefill_position_ids, kvcache, False,
                  True, None, None)

            kvcache.alloc(1)

            for j in range(decode_step):

                layer(decode_hidden[i], None, None, kvcache, False, True, None,
                      None)

        # bench
        for i in range(bench_epoch):
            kvcache.reset(batch_size)

            prefill_hidden = torch.randn(
                (batch_size, prefill_len, hidden_size),
                dtype=dtype,
                device=device).view(-1, hidden_size)

            decode_hidden = [
                torch.randn((batch_size, 1, hidden_size),
                            dtype=dtype,
                            device=device).view(-1, hidden_size)
                for _ in range(decode_step)
            ]

            torch.cuda.synchronize()
            tic = time.time()
            kvcache.alloc(prefill_len)
            layer(prefill_hidden, None, prefill_position_ids, kvcache, False,
                  True, None, None)
            torch.cuda.synchronize()
            toc = time.time()
            prefill_time.append(toc - tic)

            torch.cuda.synchronize()
            tic = time.time()
            kvcache.alloc(1)
            for j in range(decode_step):
                layer(decode_hidden[i], None, None, kvcache, False, True, None,
                      None)
            torch.cuda.synchronize()
            toc = time.time()
            decode_time.append(toc - tic)

    print(
        f"prefill time cost [{prefill_len}]: {np.mean(prefill_time[2:]) * 1000} ms"
    )
    print(
        f"decode time cost [{prefill_len}, {decode_step}]: {np.mean(decode_time[2:]) * 1000} ms"
    )


if __name__ == "__main__":
    device = "cuda:6"
    torch.cuda.set_device(device)

    # decode_step = 200

    # model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    # model_path = "/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/root/data/meta-llama/Meta-Llama-3.1-8B-Instruct/"
    model_path = "/root/data/togethercomputer/LLaMA-2-7B-32K/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = 1
    config.torch_dtype = torch.float16
    config._attn_implementation = "flash_attention_2"

    layer = CustomLlamaDecoderLayer(config,
                                    0).eval().to(torch.float16).to(device)
    

    for batch_size in [4]:
        print(f"batch_size: {batch_size}")
        for prefill_len in [64000]:
            kvcache = LokiStaticCache(
                config=config,
                num_channels=32,
                device=device,
                max_gpu_cache_memory_size=12 * 1024 * 1024 * 1024,
                sparse_ratio=prefill_len * 0.016,
                num_skip_layers=0,
                aux_data_path=
                "/root/data/loki_pca/Llama-2-7B-32K-Instruct/",
            )
            kvcache.build_cache()
            bench(layer, kvcache, batch_size, prefill_len, prefill_len // 10, device)
