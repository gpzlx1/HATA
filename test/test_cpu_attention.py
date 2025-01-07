import torch
import math
import time
import KVLib as capi
import numpy as np


def test_cpu_attention():
    bsz = 1  # batch size
    num_head = 32  # number of attention heads
    num_kv_head = 8
    gqa_size = num_head // num_kv_head
    head_dim = 128  # head dimension
    scale = 1 / math.sqrt(head_dim)

    # for seq in range(1000, 20000, 1000):
    for seq in [1000, 2000, 5000, 10000]:
        print(f"sequence length: {seq}")

        n_threads = 32

        cpu_query = capi.create_tensor([bsz, num_head, 1, head_dim],
                                       32).reshape(bsz, num_head, 1, head_dim)
        cpu_key = capi.create_tensor([bsz, num_kv_head, seq + 100, head_dim],
                                     16).reshape(bsz, num_kv_head, seq + 100,
                                                 head_dim)
        cpu_value = capi.create_tensor([bsz, num_kv_head, seq + 100, head_dim],
                                       16).reshape(bsz, num_kv_head, seq + 100,
                                                   head_dim)

        time_list = []

        for _ in range(30):
            query = torch.randn((bsz, num_head, 1, head_dim),
                                dtype=torch.float32,
                                device="cuda")
            key = torch.randn((bsz, num_kv_head, seq + 100, head_dim),
                              dtype=torch.float16,
                              device="cuda")
            value = torch.randn((bsz, num_kv_head, seq + 100, head_dim),
                                dtype=torch.float16,
                                device="cuda")

            cpu_query.copy_(query)
            cpu_key.copy_(key)
            cpu_value.copy_(value)
            torch.cuda.synchronize()

            tic = time.time()
            result, lse_res = capi.cpu_attn(cpu_query, cpu_key, cpu_value,
                                            scale, seq, True, n_threads)
            toc = time.time()
            # print(f"{(toc - tic) * 1000000:.3f} us")

            time_list.append((toc - tic) * 1000000)

        print(np.mean(time_list[5:]))

        if gqa_size > 1:
            cpu_key = cpu_key.unsqueeze(2).expand(-1, -1, gqa_size, -1,
                                                  -1).reshape(
                                                      bsz, num_head, seq + 100,
                                                      head_dim)
            cpu_value = cpu_value.unsqueeze(2).expand(
                -1, -1, gqa_size, -1, -1).reshape(bsz, num_head, seq + 100,
                                                  head_dim)

        sdpa_attn, lse = torch._scaled_dot_product_flash_attention_for_cpu(
            cpu_query.to(torch.float16),
            cpu_key[:, :, :seq, :],
            cpu_value[:, :, :seq, :],
            scale=1 / math.sqrt(head_dim))

        print("sdpa attention")
        print(sdpa_attn, sdpa_attn.shape)
        print("cpu attention result")
        print(result, result.shape)
        print((sdpa_attn - result.to(torch.float16)).abs().max())

        print("sdpa lse")
        print(lse.flatten(), lse.shape)
        print("cpu attention lse")
        print(lse_res.flatten(), lse_res.shape)
        print((lse - lse_res.squeeze(1)).abs().max())


if __name__ == "__main__":
    test_cpu_attention()
