import torch
import myTransformer
import time
from functools import partial


def bench(func):
    for i in range(5):
        func()

    # torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    # torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / 100


def torch_append(cache, key, value, pos):
    cache[0, :, pos:pos + 1, :, :] = key
    cache[1, :, pos:pos + 1, :, :] = value
    return cache


b = 2
s = 100
h = 32
d = 128
dtype = torch.float16
device = "cuda:7"

kvcache1 = torch.zeros((2, b, s, h, d), dtype=dtype, device=device)
kvcache2 = torch.zeros((2, b, s, h, d), dtype=dtype, device=device)

# warmup
key = torch.randn((b, 1, h, d), dtype=dtype, device=device)
value = torch.randn((b, 1, h, d), dtype=dtype, device=device)
insert_pos = s // 4

kvcache1 = torch_append(kvcache1, key, value, insert_pos)
myTransformer.capi.kvcache_append(kvcache2, key, value, insert_pos)

# cold start
key = torch.randn((b, 1, h, d), dtype=dtype, device=device)
value = torch.randn((b, 1, h, d), dtype=dtype, device=device)
insert_pos = s // 2

datasize = key.numel() * dtype.itemsize * 4 / 1024 / 1024 / 1024

tic = time.time()
kvcache1 = torch_append(kvcache1, key, value, insert_pos)
torch_cold_start_time = time.time() - tic
print(
    f"Torch[cold-start] time: {torch_cold_start_time * 1000:.3f} ms, bandwidth: {datasize / torch_cold_start_time:.3f} GB/s"
)

tic = time.time()
myTransformer.capi.kvcache_append(kvcache2, key, value, insert_pos)
ours_cold_start_time = time.time() - tic
print(
    f"Ours[cold-start] time: {ours_cold_start_time * 1000:.3f} ms, bandwidth: {datasize / ours_cold_start_time:.3f} GB/s"
)

torch_time = bench(partial(torch_append, kvcache1, key, value, insert_pos))
print(
    f"Torch[hot] time: {torch_time * 1000:.3f} ms, bandwidth: {datasize / torch_time:.3f} GB/s"
)
ours_time = bench(
    partial(myTransformer.capi.kvcache_append, kvcache2, key, value,
            insert_pos))
print(
    f"Ours[hot] time: {ours_time * 1000:.3f} ms, bandwidth: {datasize / ours_time:.3f} GB/s"
)

assert torch.equal(kvcache1, kvcache2)
