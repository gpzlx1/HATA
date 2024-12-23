import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import myTransformer
import os
from transformers.generation.configuration_utils import GenerationConfig
from myTransformer.models.modeling_llama_hash import CustomLlamaDecoderLayer
from myTransformer.cache.kvcache_hash import HashStaticCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bench(layer: CustomLlamaDecoderLayer,
          kvcache: HashStaticCache,
          batch_size,
          prefill_len,
          decode_step,
          device,
          dtype=torch.float16):

    warmup = 3
    bench_epoch = 10
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

            # torch.cuda.synchronize()
            tic = time.time()

            kvcache.alloc(prefill_len)
            layer(prefill_hidden, None, prefill_position_ids, kvcache, False,
                  True, None, None)

            # torch.cuda.synchronize()
            toc = time.time()
            prefill_time.append(toc - tic)

            kvcache.alloc(1)

            for j in range(decode_step):
                # torch.cuda.synchronize()
                tic = time.time()

                layer(decode_hidden[i], None, None, kvcache, False, True, None,
                      None)

                # torch.cuda.synchronize()
                toc = time.time()
                decode_time.append(toc - tic)

    print(
        f"prefill time cost [{prefill_len}]: {np.mean(prefill_time) * 1000} ms"
    )
    print(
        f"decode time cost [{prefill_len}, {decode_step}]: {np.mean(decode_time) * 1000} ms"
    )


if __name__ == "__main__":
    device = "cuda:7"
    torch.cuda.set_device(device)

    decode_step = 200
    hash_rbits = 128
    sparse_ratio = 0.03

    model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    # model_path = "/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = 1
    config.torch_dtype = torch.float16
    config._attn_implementation = "flash_attention_2"

    from myTransformer.models.modeling_llama_hash import CustomLlamaDecoderLayer
    from myTransformer.cache.kvcache_hash import HashStaticCache
    layer = CustomLlamaDecoderLayer(config,
                                    0).eval().to(torch.float16).to(device)
    kvcache = HashStaticCache(config=config,
                              hash_rbits=hash_rbits,
                              device=device,
                              max_gpu_cache_memory_size=30 * 1024 * 1024 *
                              1024,
                              sparse_ratio=sparse_ratio,
                              num_skip_layers=0,
                              hash_weights_path=None,
                              use_norm=True,
                              num_sink=64,
                              num_recent=32)
    kvcache.build_cache()

    for batch_size in [1]:
        print(f"batch_size: {batch_size}")
        for prefill_len in [96000]:
            bench(layer, kvcache, batch_size, prefill_len, decode_step, device)
