import torch
import math
from KVLib import CpuAttention


def test_cpu_attention():
    bsz = 1  # batch size
    num_head = 32  # number of attention heads
    head_dim = 128  # head dimension

    # for seq in range(1000, 20000, 1000):
    for seq in [7000]:
        gather_len = int(seq * 0.1)
        print(f"sequence length:{seq}")
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

        ne = list(key.shape)  # [bsz, num_head, 1, head_dim]
        ne.reverse()

        mem_size = 2 * 1024 * 1024 * 1024  # 2 GB
        n_dims = len(ne)

        cpu_attention = CpuAttention(mem_size, n_dims, ne)

        cpu_attention.FillKeyValye(key, value)

        result = torch.zeros(bsz * num_head * 1 * (head_dim + 1),
                             dtype=torch.float32,
                             device="cpu",
                             pin_memory=True)

        for _ in range(10):
            gather_idx = torch.randperm(bsz * num_head * seq,
                                        dtype=torch.int64,
                                        device="cpu")[:bsz * num_head *
                                                      gather_len]
            gather_idx = gather_idx.view(bsz, num_head, gather_len, 1) % seq
            cpu_attention.SparseAttentionWithMeta(query, result, gather_idx)
            print()

        lse_res = result[bsz * num_head * 1 * head_dim:]
        result = result[:bsz * num_head * 1 * head_dim]
        lse_res = lse_res.view(bsz, 1, num_head, 1)
        result = result.view(bsz, 1, num_head, head_dim)

        expand_idx = gather_idx.expand(-1, -1, -1, head_dim)
        selected_key = torch.gather(key, -2, expand_idx)
        selected_value = torch.gather(value, -2, expand_idx)

        sdpa_attn, lse = torch._scaled_dot_product_flash_attention_for_cpu(
            query.to(torch.float16),
            selected_key,
            selected_value,
            scale=1 / math.sqrt(head_dim))

        print("sdpa attention")
        print(sdpa_attn, sdpa_attn.shape)
        print("cpu attention result")
        print(result, result.shape)
        print(
            (sdpa_attn.transpose(1, 2) - result.to(torch.float16)).abs().max())

        print("sdpa lse")
        print(lse.flatten(), lse.shape)
        print("cpu attention lse")
        print(lse_res.flatten(), lse_res.shape)
        print((lse - lse_res.squeeze(1)).abs().max())


if __name__ == "__main__":
    test_cpu_attention()
