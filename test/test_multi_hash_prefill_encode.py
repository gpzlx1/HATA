import torch
from myTransformer.cache.kernels.triton_hash_encode import prefill_multi_hash_encode


def torch_prefill_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
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
        coded_data[:, i, :, :] = tmp.reshape(B, S, R // 32)

    data = data.transpose(1, 2)
    data_norm = data.norm(dim=-1)
    coded_data = coded_data.transpose(1, 2)
    
    return coded_data, data_norm


B = 2
S = 12345
N = 32
D = 128
R = 128

data = torch.randn((B, S, N, D), dtype=torch.float16, device='cuda')
hash_weight = torch.randn((N, D, R), dtype=torch.float16, device='cuda')
hash_packbit_aux_tensors = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
data_code_output = torch.zeros((B, S, N, R // 32),
                               dtype=torch.int32,
                               device='cuda')
data_norm_output = torch.zeros((B, S, N), dtype=torch.float16, device='cuda')

prefill_multi_hash_encode(data, hash_weight, data_code_output, data_norm_output, hash_packbit_aux_tensors)

# torch run
torch_encoded, torch_norm = torch_prefill_hash_encode(data, hash_weight, hash_packbit_aux_tensors)

# check
print(torch.max(torch.abs(torch_norm - data_norm_output)))
print((torch_encoded != data_code_output).sum())