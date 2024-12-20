from myTransformer.cache.kernels.triton_hash_encode import decode_hash_encode, hash_encode
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

BSZ = 1
SEQ = 1
NUM_KV_HEAD = 8
HEAD = 32
HEAD_DIM = 128
RBIT = 256
INDEX = 2

key_states = torch.randn((BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=torch.device("cuda"))
quert_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
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

triton_output = torch.empty((BSZ, SEQ * 10, NUM_KV_HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=key_states.device)
triton_norm = torch.empty((BSZ, SEQ * 10, NUM_KV_HEAD),
                          dtype=key_states.dtype,
                          device=key_states.device)
q_triton_output = torch.empty((BSZ, SEQ, HEAD, int(RBIT / 32)),
                              dtype=torch.int32,
                              device=key_states.device)

torch_output, torch_norm = torch_hash_encode(key_states, hash_weight,
                                             packbit_aux_tensor)
q_torch_output, _ = torch_hash_encode(quert_states, hash_weight,
                                      packbit_aux_tensor)
capi.decode_hash_encode(key_states, hash_weight, triton_output, triton_norm,
                        quert_states, q_triton_output, packbit_aux_tensor,
                        INDEX)

# print(q_torch_output.shape)
# print(torch_output.shape)
# print(torch_norm.shape)

# print(q_triton_output.shape)
# print(triton_output.shape)
# print(triton_norm.shape)

# print(torch_output)
# print(triton_output[:, INDEX:INDEX+1, :, :])

# print(q_torch_output)
# print(q_triton_output)

# print(torch_norm)
# print(triton_norm[:, INDEX:INDEX+1, :])

assert (q_torch_output == q_triton_output).all()
assert (torch_output == triton_output[:, INDEX:INDEX + 1, :, :]).all()
print(torch.abs(torch_norm - triton_norm[:, INDEX:INDEX + 1, :]).max())

# # batched_q_k = torch.cat([quert_states, key_states], dim=0)

# bench(partial(torch_hash_encode, key_states, hash_weight, packbit_aux_tensor))

torch.cuda.nvtx.range_push("new_decode_hash_encode")
bench(
    partial(capi.decode_hash_encode, key_states, hash_weight, triton_output,
            triton_norm, quert_states, q_triton_output, packbit_aux_tensor,
            INDEX))
torch.cuda.nvtx.range_pop()

# torch.cuda.nvtx.range_push("hash_encode_key_states")
# bench(partial(hash_encode, key_states, hash_weight, packbit_aux_tensor))
# torch.cuda.nvtx.range_pop()

# # bench(partial(hash_encode, batched_q_k, hash_weight, packbit_aux_tensor))
# bench(partial(matmul, key_states, hash_weight))
