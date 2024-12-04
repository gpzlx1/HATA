from myTransformer.cache.kernels.triton_hash_encode import hash_encode
import torch
from functools import partial
from myTransformer import capi


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


@torch.compile(fullgraph=True)
def torch_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                      packbit_aux_tensor: torch.Tensor):
    output_dtype = torch.int32
    chunk_size = 32

    RBIT = hash_weight.shape[1]
    chunk_num = int(RBIT / chunk_size)

    if len(data.data.shape) == 3:
        BSZ, HEAD, HEAD_DIM = data.shape
        output_shape = (BSZ, HEAD, chunk_num, chunk_size)
    else:
        BSZ, SEQ, HEAD, HEAD_DIM = data.shape
        output_shape = (BSZ, SEQ, HEAD, chunk_num, chunk_size)

    key_code = torch.matmul(data, hash_weight) > 0
    packbit_key_code = key_code.reshape(*output_shape)
    packbit_key_code = packbit_key_code * packbit_aux_tensor
    packbit_key_code = packbit_key_code.sum(dim=-1, dtype=output_dtype)

    return packbit_key_code


def cuda_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor):
    return capi.encode(data, hash_weight)


torch.cuda.set_device(2)
torch.manual_seed(42)
torch.set_printoptions(precision=2)

SEQ = 1
HEAD = 128
HEAD_DIM = 128
RBIT = 256

key_states = torch.randn((SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=torch.device("cuda"))

key_states = (key_states * 5).int().to(torch.float16)

hash_weight = torch.normal(
    10,
    100,
    size=(RBIT, HEAD_DIM),
    device=key_states.device,
    dtype=torch.float16,
).int().to(torch.float16)
packbit_aux_tensor = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))

torch_output = torch_hash_encode(key_states, hash_weight.T.contiguous(),
                                 packbit_aux_tensor)
# print(torch_output)

triton_output = hash_encode(key_states, hash_weight.T.contiguous(),
                            packbit_aux_tensor)
# print(triton_output)

assert (torch_output == triton_output).all()

cuda_output = cuda_hash_encode(key_states.view(-1, HEAD_DIM), hash_weight)

print((torch_output == cuda_output).sum(), torch_output.numel())

bench((partial(torch_hash_encode, key_states, hash_weight.T.contiguous(),
               packbit_aux_tensor)))
bench(
    partial(hash_encode, key_states, hash_weight.T.contiguous(),
            packbit_aux_tensor))
bench(partial(cuda_hash_encode, key_states.view(-1, HEAD_DIM), hash_weight))
bench(partial(torch.matmul, key_states, hash_weight.T.contiguous()))
