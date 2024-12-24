import torch
import math
import time
from KVLib import CpuAttention


def test_cpu_attention():
    bsz = 1  # batch size
    num_head = 32  # number of attention heads
    head_dim = 128  # head dimension
    scale = 1 / math.sqrt(head_dim)

    # for seq in range(1000, 20000, 1000):
    for seq in [1000]:
        print(f"sequence length: {seq}")

        mem_size = 2 * 1024 * 1024 * 1024  # 2 GB
        n_threads = 64

        cpu_attention = CpuAttention(mem_size, n_threads)

        for _ in range(10):
            query = torch.randn((bsz, num_head, 1, head_dim),
                                dtype=torch.float32,
                                device="cpu",
                                pin_memory=True)
            key = torch.randn((bsz, num_head, seq, head_dim),
                              dtype=torch.float16,
                              device="cpu",
                              pin_memory=True)
            value = torch.randn((bsz, num_head, seq, head_dim),
                                dtype=torch.float16,
                                device="cpu",
                                pin_memory=True)
            tic = time.time()
            result = cpu_attention.Attention(query, key, value, scale)
            toc = time.time()
            print(f"{(toc - tic) * 1000000:.3f} us")

        sdpa_attn, _ = torch._scaled_dot_product_flash_attention_for_cpu(
            query.to(torch.float16), key, value, scale=scale)

        print("sdpa attention")
        print(sdpa_attn, sdpa_attn.shape)
        print("cpu attention result")
        print(result, result.shape)
        print(
            (sdpa_attn.transpose(1, 2) - result.to(torch.float16)).abs().max())


if __name__ == "__main__":
    test_cpu_attention()
