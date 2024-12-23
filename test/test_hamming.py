import torch
import myTransformer
from functools import partial
from myTransformer.cache.kernels.triton_hash_encode import hash_encode

torch.cuda.set_device(6)
torch.manual_seed(42)


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
    latency = (t1 - t0) / 100
    print(latency * 1000)
    return latency


def torch_hamming_distance(key, query, hash_weight):
    h = query.shape[2]
    b, s, hk, _ = key.shape
    gqa = h // hk
    rbit = hash_weight.shape[-1]
    key = torch.matmul(key, hash_weight) > 0
    query = torch.matmul(query, hash_weight) > 0
    key = key.view(b, s, hk, 1, rbit).expand(-1, -1, -1, gqa,
                                             -1).reshape(b, s, h, rbit)
    hamming_distance = ((query.to(torch.float16) -
                         key.to(torch.float16)).abs().sum(dim=-1))
    return hamming_distance


def encode(key, query, hash_weight):
    packbit_aux_tensor = torch.pow(
        2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    return key_code, query_code


b = 2
rbit = 128
h = 32
hk = 8
gqa = h // hk
s = 128000
key = torch.randn(b, s, hk, 128).to(torch.float16).cuda()
query = torch.randn(b, 1, h, 128).to(torch.float16).cuda()

hash_weight = torch.normal(
    0,
    2,
    size=(128, rbit),
    device=key.device,
    dtype=key.dtype,
)
key_norms = key.norm(dim=-1)
key_norms_expand = key_norms.view(b, s, hk, 1).expand(b, s, hk,
                                                      gqa).reshape(b, s, h)

torch_output = torch_hamming_distance(key, query, hash_weight)
torch_output = (1.0 - 2.0 * torch_output / rbit) * key_norms_expand
torch_output = torch_output.transpose(1, 2).view(b, hk, gqa, -1).sum(2)
print(torch_output)
print(torch_output.shape)

key_code, query_code = encode(key, query, hash_weight)

torch.cuda.synchronize()

my_output2 = myTransformer.capi.hamming_score(key_code,
                                              query_code,
                                              key_norms,
                                              rbit,
                                              s,
                                              use_key_norm=True)
print(my_output2)
print(my_output2.shape)

print((my_output2 - torch_output).abs().max())

latency = bench(
    partial(myTransformer.capi.hamming_score,
            key_code,
            query_code,
            key_norms,
            rbit,
            s,
            use_key_norm=True))

size = key_code.numel() * key_code.dtype.itemsize + query_code.numel(
) * query_code.dtype.itemsize + key_norms.numel(
) * key_norms.dtype.itemsize + my_output2.numel() * my_output2.dtype.itemsize
bandwidth = size / 1024 / 1024 / 1024 / latency
print(
    f"Data: {size / 1024 / 1024 / 1024:.3f} GB Bandwidth: {bandwidth:.3f} GB/sec"
)
