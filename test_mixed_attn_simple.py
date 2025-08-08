import torch
import KVLib as capi
from flash_attn import flash_attn_with_kvcache
import math
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


B = 1
QH = 8
KH = 4
SEQ_LEN = 12345
INDEX_LEN = 100
DIM = 128
sacle = 1.0 / math.sqrt(DIM)

assert KH == 4

torch.manual_seed(42)
device = torch.device('cuda:7')
dtype = torch.float16

torch.cuda.set_device(device)

query = torch.randn(B, 1, QH, DIM, device=device, dtype=dtype)
top_index = torch.randint(0, SEQ_LEN, (B, KH, INDEX_LEN),
                          device=device, dtype=torch.int64)

# Head Mask

# True for cached_keys/cached_values
# False for buffer_keys/buffer_values
k_head_mask = torch.tensor([True, True, False, False], device=device)
k_head_index = torch.tensor([0, 1, 0, 1], device=device)


# cache_keys
cached_keys = torch.randn(B, SEQ_LEN, 2, DIM, device=device, dtype=dtype)
cached_values = torch.randn(B, SEQ_LEN, 2, DIM, device=device, dtype=dtype)

gather_topk = top_index[:, 0:2, :]

# print(gather_topk)
# print(top_index)

# gather the key and value tensors using the top_index
gather_keys = torch.gather(
    cached_keys, 1, gather_topk.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))
gather_values = torch.gather(
    cached_values, 1, gather_topk.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))

buffer_keys = torch.randn(B, SEQ_LEN, 2, DIM, device=device, dtype=dtype)
buffer_values = torch.randn(B, SEQ_LEN, 2, DIM, device=device, dtype=dtype)

print(gather_keys.shape, gather_values.shape)
print(buffer_keys.shape, buffer_values.shape)

# flash attn
# attn1
flash_attn1 = flash_attn_with_kvcache(
    query[:, :, 0:4, :],
    gather_keys,
    gather_values,
)

flash_attn2 = flash_attn_with_kvcache(
    query[:, :, 4:8, :],
    buffer_keys[:, :INDEX_LEN, :, :],
    buffer_values[:, :INDEX_LEN, :, :],
)

flash_out = torch.cat([flash_attn1, flash_attn2], dim=2)

print(flash_attn1.shape)
print(flash_attn2.shape)
print(flash_out)


REAL_SEQ_LEN = INDEX_LEN
capi_attn_output, _ = capi.flash_mixed_decode(
    query, cached_keys, cached_values, top_index.int(
    ), buffer_keys, buffer_values, k_head_mask, k_head_index.int(), REAL_SEQ_LEN,
    sacle)

print(capi_attn_output)


assert torch.allclose(flash_out,
                      capi_attn_output, atol=1e-2, rtol=1e-2)