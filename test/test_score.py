import torch
import myTransformer
from functools import partial
from myTransformer.cache.kernels.triton_hash_encode import hash_encode
from myTransformer.cache.kernels.triton_score_process import hash_score_process

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


def torch_score(key, query, hash_weight):
    key_norm = key.norm(dim=-1)
    key = torch.matmul(key, hash_weight) > 0
    query = torch.matmul(query, hash_weight) > 0
    hamming_distance = ((query.to(torch.float16) -
                         key.to(torch.float16)).abs().sum(dim=-1))
    rbit = hash_weight.shape[-1]
    score = (1 - 2 * hamming_distance / rbit) * key_norm
    return score.transpose(-1, -2).contiguous()


def my_score(key, query, hash_weight, packbit_aux_tensor):
    key_norm = key.norm(dim=-1)
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    hamming_distance = myTransformer.capi.hamming_distance(
        key_code, query_code)
    output = hash_score_process(hamming_distance, key_norm,
                                hash_weight.shape[-1])
    return output


def my_hamming_torch_score(key, query, hash_weight, packbit_aux_tensor):
    key_norm = key.norm(dim=-1)
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    hamming_distance = myTransformer.capi.hamming_distance(
        key_code, query_code)
    rbit = hash_weight.shape[-1]
    score = (1 - 2 * hamming_distance / rbit) * key_norm
    return score.transpose(-1, -2).contiguous()


key = torch.randn(1, 32000, 32, 128).to(torch.float16).cuda()
query = torch.randn(1, 1, 32, 128).to(torch.float16).cuda()

hash_weight = torch.normal(
    0,
    2,
    size=(128, 256),
    device=key.device,
    dtype=key.dtype,
)
packbit_aux_tensor = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))

output = torch_score(key, query, hash_weight)
print(output)

my_output = my_score(key, query, hash_weight, packbit_aux_tensor)
torch.cuda.synchronize()
print(my_output)

diff = output - my_output
print(diff.abs().max())

bench(partial(torch_score, key, query, hash_weight))
bench(
    partial(my_hamming_torch_score, key, query, hash_weight,
            packbit_aux_tensor))
bench(partial(my_score, key, query, hash_weight, packbit_aux_tensor))
