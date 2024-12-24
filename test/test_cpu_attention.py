import torch
import math
from KVLib import CpuAttention


def test_cpu_attention():
    bsz = 1  # batch size
    num_head = 32  # number of attention heads
    head_dim = 128  # head dimension

    # for seq in range(1000, 20000, 1000):
    for seq in [7000]:
        print(f"sequence length: {seq}")

        ne = list([bsz, num_head, seq,
                   head_dim])  # [bsz, num_head, 1, head_dim]
        ne.reverse()

        mem_size = 2 * 1024 * 1024 * 1024  # 2 GB
        n_dims = len(ne)

        cpu_attention = CpuAttention(mem_size, n_dims, ne)

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
            cpu_attention.FillKeyValye(key, value)
            result = torch.zeros((bsz, 1, num_head, head_dim),
                                 dtype=torch.float32,
                                 device="cpu",
                                 pin_memory=True)
            cpu_attention.Attention(query, result)
            print()

        sdpa_attn, _ = torch._scaled_dot_product_flash_attention_for_cpu(
            query.to(torch.float16), key, value, scale=1 / math.sqrt(head_dim))

        print("sdpa attention")
        print(sdpa_attn, sdpa_attn.shape)
        print("cpu attention result")
        print(result, result.shape)
        print(
            (sdpa_attn.transpose(1, 2) - result.to(torch.float16)).abs().max())


if __name__ == "__main__":
    test_cpu_attention()
