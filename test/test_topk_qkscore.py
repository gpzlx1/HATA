import torch
from functools import partial
from myTransformer.cache.kernels.triton_qk_score import topk_qk_score

bsz = 2
num_heads = 32
num_kv_heads = 8
seq_len = 100000
head_dim = 128
partial_dim = 32

device = "cuda:0"
dtype = torch.float16

query = torch.randn(
    (bsz, 1, num_heads, head_dim),
    dtype=torch.float16,
    device=device,
)
key = torch.randn(
    (bsz, seq_len + 1000, num_kv_heads, head_dim),
    dtype=torch.float16,
    device=device,
)


def torch_topk_qk_score(query, key, seq_len):
    b, _, hk, d = key.shape
    h = query.shape[-2]
    gqa = h // hk

    query = query.transpose(1, 2).transpose(-1, -2)
    key = key[:, :seq_len, :, :].transpose(1, 2).unsqueeze(2).expand(
        -1, -1, gqa, -1, -1).reshape(b, h, seq_len, d)
    score = key @ query
    score = score.view(b, hk, gqa, -1).sum(2)
    return score


torch_out = torch_topk_qk_score(query, key, seq_len)
triton_out = topk_qk_score(query, key, seq_len)

print(torch_out)
print(torch_out.shape)
print(triton_out)
print(triton_out.shape)

print((torch_out - triton_out).abs().max())


def bench(func):
    import time

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


bench(partial(torch_topk_qk_score, query, key, seq_len))
bench(partial(topk_qk_score, query, key, seq_len))
