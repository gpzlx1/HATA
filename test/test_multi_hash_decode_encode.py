import torch
from myTransformer.cache.kernels.triton_hash_encode import decode_multi_hash_encode

torch.manual_seed(12)


def torch_key_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                          packbit_tensor: torch.Tensor):

    b, s, n, d = data.shape
    r = hash_weight.shape[-1]
    data = data.transpose(1, 2)  # [B, H, S, D]
    coded_data = torch.zeros((b, n, s, r // 32),
                             dtype=torch.int32,
                             device='cuda')

    for i in range(n):
        tmp = torch.matmul(data[:, i, :, :], hash_weight[i, :, :])
        tmp = tmp > 0
        tmp = tmp.reshape(b, s, r // 32, 32)
        tmp = tmp.to(packbit_tensor.dtype)
        tmp = tmp * packbit_tensor
        tmp = tmp.sum(dim=-1)
        tmp = tmp.reshape(b, s, r // 32)
        coded_data[:, i, :, :] = tmp.reshape(b, s, r // 32)

    data = data.transpose(1, 2)
    data_norm = data.norm(dim=-1)
    coded_data = coded_data.transpose(1, 2)

    return coded_data, data_norm


def torch_query_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                            packbit_tensor: torch.Tensor):

    b, s, n, d = data.shape
    r = hash_weight.shape[-1]
    m = hash_weight.shape[0]

    kv_group = n // m

    data = data.transpose(1, 2)  # [B, H, S, D]
    coded_data = torch.zeros((b, n, s, r // 32),
                             dtype=torch.int32,
                             device='cuda')

    for i in range(m):
        tmp = data[:, i * kv_group:(i + 1) * kv_group, :, :]
        tmp = torch.matmul(tmp, hash_weight[i, :, :])
        tmp = tmp > 0
        tmp = tmp.reshape(b, kv_group, s, r // 32, 32)
        tmp = tmp.to(packbit_tensor.dtype)
        tmp = tmp * packbit_tensor
        tmp = tmp.sum(dim=-1)
        tmp = tmp.reshape(b, kv_group, s, r // 32)
        coded_data[:, i * kv_group:(i + 1) * kv_group, :, :] = tmp.reshape(
            b, kv_group, s, r // 32)

    data = data.transpose(1, 2)
    data_norm = data.norm(dim=-1)
    coded_data = coded_data.transpose(1, 2)

    return coded_data, data_norm


B = 32
S = 1
NUM_HEAD = 32
NUM_KV_HEAD = 8
D = 128
R = 256

query = torch.randn((B, S, NUM_HEAD, D), dtype=torch.float16, device='cuda')
key = torch.randn((B, S, NUM_KV_HEAD, D), dtype=torch.float16, device='cuda')
hash_weight = torch.randn((NUM_KV_HEAD, D, R),
                          dtype=torch.float16,
                          device='cuda')
hash_packbit_aux_tensors = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))

# print(query)

query_encoded = torch.zeros((B, S, NUM_HEAD, R // 32),
                            dtype=torch.int32,
                            device='cuda')
key_encoded = torch.zeros((B, S + 1, NUM_KV_HEAD, R // 32),
                          dtype=torch.int32,
                          device='cuda')
key_norm = torch.zeros((B, S + 1, NUM_KV_HEAD),
                       dtype=torch.float16,
                       device='cuda')

decode_multi_hash_encode(key, hash_weight, key_encoded, key_norm, query,
                         query_encoded, hash_packbit_aux_tensors, 1)

# print(key_norm)

torch_key_encoded, torch_key_norm = torch_key_hash_encode(
    key, hash_weight, hash_packbit_aux_tensors)

# print(torch_key_norm)

torch_query_encoded, _ = torch_query_hash_encode(query, hash_weight,
                                                 hash_packbit_aux_tensors)

# print(query_encoded)
# print(torch_query_encoded)
# print(torch_query_norm)

# print(key_encoded)
# print(torch_key_encoded)

# print(key_norm)
# print(torch_key_norm)

# check
print((query_encoded != torch_query_encoded).sum())

# check
print((key_encoded[:, 1:2] != torch_key_encoded).sum())

# check
print(torch.max(torch.abs(key_norm[:, 1:2] - torch_key_norm)))
