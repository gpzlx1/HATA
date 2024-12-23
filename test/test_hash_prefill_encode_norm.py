from myTransformer.cache.kernels.triton_hash_encode import prefill_hash_encode, hash_encode
import torch
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


def matmul(key_states, hash_weight):
    return torch.matmul(key_states, hash_weight)


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

    return packbit_key_code, key_states.norm(dim=-1)


torch.cuda.set_device(7)
torch.manual_seed(42)

BSZ = 10
SEQ = 8000
HEAD = 32
HEAD_DIM = 128
RBIT = 256

key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=torch.device("cuda"))

hash_weight = torch.normal(
    0,
    2,
    size=(HEAD_DIM, RBIT),
    device=key_states.device,
    dtype=key_states.dtype,
)
packbit_aux_tensor = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))

triton_output = torch.empty((BSZ, SEQ * 2, HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=key_states.device)
triton_norm = torch.empty((BSZ, SEQ * 2, HEAD),
                          dtype=key_states.dtype,
                          device=key_states.device)

torch_output, torch_norm = torch_hash_encode(key_states, hash_weight,
                                             packbit_aux_tensor)
prefill_hash_encode(key_states, hash_weight, triton_output, triton_norm,
                    packbit_aux_tensor)

# print(torch_output)
# print(triton_output[:, :SEQ, :, :])

# print(torch_norm)
# print(triton_norm[:, :SEQ, :])

assert (torch_output == triton_output[:, :SEQ, :, :]).all()
print(torch.abs(torch_norm - triton_norm[:, :SEQ, :]).max())

bench(partial(torch_hash_encode, key_states, hash_weight, packbit_aux_tensor))
bench(
    partial(prefill_hash_encode, key_states, hash_weight, triton_output,
            triton_norm, packbit_aux_tensor))
bench(partial(hash_encode, key_states, hash_weight, packbit_aux_tensor))
bench(partial(matmul, key_states, hash_weight))
