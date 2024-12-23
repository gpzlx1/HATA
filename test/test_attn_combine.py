import torch
import math
from flash_attn import flash_attn_func
import myTransformer


def flashattn(q, k, v, scale):
    attn, lse, _ = flash_attn_func(q,
                                   k,
                                   v,
                                   softmax_scale=scale,
                                   return_attn_probs=True)
    return attn, lse


b = 2
h = 32
hk = 8
n = 16000
d = 128

n1 = 96
n2 = n - n1

dtype = torch.float16
device = "cuda:0"
scale = 1 / math.sqrt(d)
scale_log2 = scale * math.log2(math.e)

query = torch.randn((b, 1, h, d), device=device, dtype=dtype)
key = torch.randn((b, n, hk, d), device=device, dtype=dtype)
value = torch.randn((b, n, hk, d), device=device, dtype=dtype)

key1 = key[:, :n1, :, :]
value1 = value[:, :n1, :, :]

key = key[:, n1:, :, :]
value = value[:, n1:, :, :]

gather_len = 3000

gather_idx = torch.randperm(b * hk * n2,
                            dtype=torch.int32,
                            device=torch.device("cuda"))[:b * hk * gather_len]
gather_idx = gather_idx.view(b, hk, gather_len) % n2

# ref
gather_idx_long = gather_idx.transpose(-1,
                                       -2).unsqueeze(-1).expand(-1, -1, -1,
                                                                d).long()
sub_key = torch.gather(key, 1, gather_idx_long)
sub_value = torch.gather(value, 1, gather_idx_long)
sub_key = torch.cat([key1, sub_key], dim=1)
sub_value = torch.cat([value1, sub_value], dim=1)
attn_ref, _ = myTransformer.models.utils.flash_attnention(
    query, sub_key, sub_value, scale)

# ours
attn1, lse1 = myTransformer.models.utils.flash_attnention(
    query, key1, value1, scale)
attn2, lse2 = myTransformer.capi.flash_index_decode(query, key, value,
                                                    gather_idx, scale)

attn = myTransformer.capi.combine_attention(attn1, lse1, attn2, lse2)

print(attn)
print(attn_ref)
print((attn - attn_ref).abs().max())

from functools import partial


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


bench(
    partial(myTransformer.models.utils.combine_attention, [attn1, attn2],
            [lse1, lse2]))
bench(partial(myTransformer.capi.combine_attention, attn1, lse1, attn2, lse2))
