import torch
import math
import time
from KVLib import CpuAttention
import KVLib as capi
import numpy as np


def test_cpu_attention():
    bsz = 1  # batch size
    num_head = 32  # number of attention heads
    head_dim = 128  # head dimension
    scale = 1 / math.sqrt(head_dim)

    # for seq in range(1000, 20000, 1000):
    for seq in [10000, 20000, 50000, 100000]:
        gather_len = int(seq * 0.1)
        print(f"sequence length: {seq}, gather: {gather_len}")

        mem_size = 2 * 1024 * 1024 * 1024  # 2 GB
        n_threads = 32

        cpu_attention = CpuAttention(mem_size, n_threads)
        # cpu_query = torch.empty((bsz, num_head, 1, head_dim),
        #                         dtype=torch.float32,
        #                         device="cpu", pin_memory=True)
        # cpu_key = torch.empty((bsz, num_head, seq, head_dim),
        #                       dtype=torch.float16,
        #                       device="cpu", pin_memory=True)
        # cpu_value = torch.empty((bsz, num_head, seq, head_dim),
        #                         dtype=torch.float16,
        #                         device="cpu", pin_memory=True)

        cpu_query = capi.create_tensor([bsz, num_head, 1, head_dim],
                                       32).reshape(bsz, num_head, 1, head_dim)
        cpu_key = capi.create_tensor([bsz, num_head, seq, head_dim],
                                     16).reshape(bsz, num_head, seq, head_dim)
        cpu_value = capi.create_tensor([bsz, num_head, seq, head_dim],
                                       16).reshape(bsz, num_head, seq,
                                                   head_dim)

        time_list = []

        for _ in range(30):
            query = torch.randn((bsz, num_head, 1, head_dim),
                                dtype=torch.float32,
                                device="cuda")
            key = torch.randn((bsz, num_head, seq, head_dim),
                              dtype=torch.float16,
                              device="cuda")
            value = torch.randn((bsz, num_head, seq, head_dim),
                                dtype=torch.float16,
                                device="cuda")

            # query = query.cpu()
            # key = key.cpu()
            # value = value.cpu()
            cpu_query.copy_(query)
            cpu_key.copy_(key)
            cpu_value.copy_(value)
            torch.cuda.synchronize()

            gather_idx = torch.randperm(bsz * num_head * seq,
                                        dtype=torch.int32,
                                        device="cuda")[:bsz * num_head *
                                                       gather_len]
            gather_idx = gather_idx.view(bsz, num_head, gather_len, 1) % seq
            gather_idx = gather_idx.cpu()
            torch.cuda.synchronize()

            tic = time.time()
            result = cpu_attention.SparseAttention(cpu_query,
                                                   cpu_key, cpu_value,
                                                   gather_idx.int(), scale)
            toc = time.time()
            # print(f"{(toc - tic) * 1000000:.3f} us")

            time_list.append((toc - tic) * 1000000)

        print(np.mean(time_list[5:]))

        # print(gather_idx.shape)

        # expand_idx = gather_idx.expand(-1, -1, -1, head_dim).long()
        # selected_key = torch.gather(key.cpu(), -2, expand_idx)
        # selected_value = torch.gather(value.cpu(), -2, expand_idx)

        # sdpa_attn, _ = torch._scaled_dot_product_flash_attention_for_cpu(
        #     query.cpu().to(torch.float16),
        #     selected_key,
        #     selected_value,
        #     scale=1 / math.sqrt(head_dim))

        # print("sdpa attention")
        # print(sdpa_attn, sdpa_attn.shape)
        # print("cpu attention result")
        # print(result, result.shape)
        # print(
        #     (sdpa_attn.transpose(1, 2) - result.to(torch.float16)).abs().max())


if __name__ == "__main__":
    test_cpu_attention()
