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


B = 4
QH = 32
KH = 8
SEQ_LEN = 32000
INDEX_LEN = 20000
DIM = 128
sacle = 1.0 / math.sqrt(DIM)

assert KH == 8

torch.manual_seed(42)
device = torch.device('cuda:6')
dtype = torch.float16

torch.cuda.set_device(device)

query = torch.randn(B, 1, QH, DIM, device=device, dtype=dtype)
top_index = torch.randint(0, SEQ_LEN, (B, KH, INDEX_LEN),
                          device=device, dtype=torch.int64)

# Head Mask

# True for cached_keys/cached_values
# False for buffer_keys/buffer_values
k_head_mask = torch.tensor(
    [False, True, True, True, False, False, False, True], device=device)
q_head_mask = k_head_mask.repeat_interleave(QH // KH)
k_head_index = torch.tensor([0, 0, 1, 2, 1, 2, 3, 3], device=device)
gather_topk = top_index[:, k_head_mask, :]
# print(gather_topk)
# print(top_index)

# cache_keys
cached_keys = torch.randn(B, SEQ_LEN, 4, DIM, device=device, dtype=dtype)
cached_values = torch.randn(B, SEQ_LEN, 4, DIM, device=device, dtype=dtype)
buffer_keys = torch.randn(B, SEQ_LEN, 4, DIM, device=device, dtype=dtype)
buffer_values = torch.randn(B, SEQ_LEN, 4, DIM, device=device, dtype=dtype)


# gather the key and value tensors using the top_index
gather_keys = torch.gather(
    cached_keys, 1, gather_topk.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))
gather_values = torch.gather(
    cached_values, 1, gather_topk.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))


print(gather_keys.shape, gather_values.shape)
print(buffer_keys.shape, buffer_values.shape)

# flash attn
# attn1
flash_cached_attn = flash_attn_with_kvcache(
    query[:, :, q_head_mask, :],
    gather_keys,
    gather_values,
)

flash_buffer_attn = flash_attn_with_kvcache(
    query[:, :, ~q_head_mask, :],
    buffer_keys[:, :INDEX_LEN, :, :],
    buffer_values[:, :INDEX_LEN, :, :],
)


print(flash_cached_attn.shape)
print(flash_buffer_attn.shape)


REAL_SEQ_LEN = INDEX_LEN
capi_attn_output, _ = capi.flash_mixed_decode(
    query, cached_keys, cached_values, top_index.int(
    ), buffer_keys, buffer_values, k_head_mask, k_head_index.int(), REAL_SEQ_LEN,
    sacle)


assert torch.allclose(flash_cached_attn,
                      capi_attn_output[:, :, q_head_mask], atol=1e-2, rtol=1e-2)
assert torch.allclose(flash_buffer_attn,
                      capi_attn_output[:, :, ~q_head_mask], atol=1e-2, rtol=1e-2)


# # # bench mark
all_keys = torch.cat([gather_keys, gather_keys], dim=-2)
all_values = torch.cat([gather_values, gather_values], dim=-2)

bench((partial(flash_attn_with_kvcache, query, all_keys, all_values,)))

bench((partial(capi.flash_index_decode, query, torch.cat([cached_keys, cached_keys], dim=-2), torch.cat([cached_keys, cached_keys], dim=-2), top_index.int(
), sacle,)))

bench((partial(capi.flash_mixed_decode, query, cached_keys, cached_values, top_index.int(
), buffer_keys, buffer_values, k_head_mask, k_head_index.int(), REAL_SEQ_LEN,
    sacle,)))
