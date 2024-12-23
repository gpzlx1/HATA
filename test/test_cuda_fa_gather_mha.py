import torch
from functools import partial
import myTransformer
import math


def bench(func):
    import time
    import numpy as np

    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) * 1000 / 100)


def gather_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                idx: torch.Tensor):
    d = q.shape[-1]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
    gather_k = torch.gather(k, -2, idx)
    gather_v = torch.gather(v, -2, idx)
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q, gather_k, gather_v, attn_mask=None, dropout_p=0.0, is_causal=False)

    return attn_out.transpose(1, 2).contiguous()


def gather_kv(k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor):
    d = k.shape[-1]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
    gather_k = torch.gather(k, -2, idx)
    gather_v = torch.gather(v, -2, idx)

    return gather_k, gather_v


torch.cuda.set_device(0)
torch.manual_seed(42)

batch_size = 2
num_heads = 32
num_kv_heads = 32
head_dim = 128
seq_len = 32000
gather_lens = [100, 500, 1000, 2000, 4000, 8000, 16000]

print(f"Batchsize {batch_size}, Seqlen {seq_len}")

query_states = torch.randn((batch_size, 1, num_heads, head_dim),
                           dtype=torch.float16,
                           device=torch.device("cuda"))
key_states = torch.randn((batch_size, seq_len, num_kv_heads, head_dim),
                         dtype=torch.float16,
                         device=torch.device("cuda"))
value_states = torch.randn((batch_size, seq_len, num_kv_heads, head_dim),
                           dtype=torch.float16,
                           device=torch.device("cuda"))
scale = 1 / math.sqrt(head_dim)

for gather_len in gather_lens:
    print("\ngather_len: ", gather_len)
    gather_idx = torch.randperm(
        batch_size * num_kv_heads * seq_len,
        dtype=torch.int32,
        device=torch.device("cuda"))[:batch_size * num_kv_heads * gather_len]
    gather_idx = gather_idx.view(batch_size, num_kv_heads,
                                 gather_len) % seq_len
    gather_idx_long = gather_idx.long()

    attn_out1 = gather_sdpa(query_states, key_states, value_states,
                            gather_idx_long)

    attn_out2, _ = myTransformer.capi.flash_index_decode(
        query_states, key_states, value_states, gather_idx, scale)

    print("Max diff:", (attn_out1 - attn_out2).abs().max())

    print("torch.gather: ", end="")
    bench((partial(gather_kv, key_states, value_states, gather_idx_long)))

    print("Sdpa + torch.gather: ", end="")
    bench((partial(gather_sdpa, query_states, key_states, value_states,
                   gather_idx_long)))

    print("FA + gather fusion: ", end="")
    bench((partial(myTransformer.capi.flash_index_decode, query_states,
                   key_states, value_states, gather_idx, scale)))
