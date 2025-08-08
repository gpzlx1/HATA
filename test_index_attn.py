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


B = 2
QH = 32
KH = 8
SEQ_LEN = 12345
INDEX_LEN = 1200
DIM = 128

torch.manual_seed(42)
device = torch.device('cuda:7')
dtype = torch.float16

torch.cuda.set_device(device)

query = torch.randn(B, 1, QH, DIM, device=device, dtype=dtype)
key = torch.randn(B, SEQ_LEN, KH, DIM, device=device, dtype=dtype)
value = torch.randn(B, SEQ_LEN, KH, DIM, device=device, dtype=dtype)
top_index = torch.randint(0, SEQ_LEN, (B, KH, INDEX_LEN),
                          device=device, dtype=torch.int64)


# gather the key and value tensors using the top_index
gather_keys = torch.gather(
    key, 1, top_index.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))
gather_values = torch.gather(
    value, 1, top_index.unsqueeze(-1).expand(-1, -1, -1, DIM).transpose(1, 2))

print(gather_keys.shape, gather_values.shape)

flash_gather_index_out = flash_attn_with_kvcache(
    query,
    gather_keys,
    gather_values,
)

print(flash_gather_index_out)

sacle = 1.0 / math.sqrt(DIM)
# Using capi.index_attn to compute the attention
capi_attn_output, _ = capi.flash_index_decode(
    query, key, value, top_index.int(),
    sacle)

print(capi_attn_output)

assert torch.allclose(flash_gather_index_out,
                      capi_attn_output, atol=1e-2, rtol=1e-2)


# bench mark
bench((partial(flash_attn_with_kvcache, query, gather_keys, gather_values)))

bench((partial(capi.flash_index_decode, query, key, value, top_index.int(),
               sacle)))
