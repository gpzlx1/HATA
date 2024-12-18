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
    print((t1 - t0) * 1000 / 100)


def torch_hamming_distance(key, query, hash_weight):
    key = torch.matmul(key, hash_weight) > 0
    query = torch.matmul(query, hash_weight) > 0
    hamming_distance = ((query.to(torch.float16) -
                         key.to(torch.float16)).abs().sum(dim=-1))
    return hamming_distance


def encode(key, query, hash_weight):
    packbit_aux_tensor = torch.pow(
        2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    return key_code, query_code


key = torch.randn(1, 128000, 32, 128).to(torch.float16).cuda()

query = torch.randn(1, 1, 32, 128).to(torch.float16).cuda()
rbit = 128

hash_weight = torch.normal(
    0,
    2,
    size=(128, rbit),
    device=key.device,
    dtype=key.dtype,
)
key_norms = key.norm(dim=-1)

torch_output = torch_hamming_distance(key, query, hash_weight)
torch_output = (1.0 - 2.0 * torch_output / rbit) * key_norms
torch_output = torch_output.transpose(1, 2)
print(torch_output)

key_code, query_code = encode(key, query, hash_weight)

torch.cuda.synchronize()

print(key_code.shape)
print(key_code.stride())
print(key_norms.shape)
print(key_norms.stride())

my_output2 = myTransformer.capi.hamming_score(key_code, query_code, key_norms,
                                              rbit, 128000)
print(my_output2)

print((my_output2 == torch_output).all())

bench(
    partial(myTransformer.capi.hamming_score, key_code, query_code, key_norms,
            rbit, 128000))
