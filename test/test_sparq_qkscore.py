import torch
from functools import partial
from myTransformer.cache.kernels.triton_qk_score import sparq_qk_score
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
bsz = 2
num_heads = 32
num_kv_heads = 8
seq_len = 100000
head_dim = 128
r_channel = 32

device = "cuda:0"
dtype = torch.float16

query = torch.randn(
    (bsz, num_heads, 1, head_dim),
    dtype=torch.float16,
    device=device,
)
key = torch.randn(
    (bsz, num_kv_heads, seq_len, head_dim),
    dtype=torch.float16,
    device=device,
)

channel_index = torch.randperm(
    bsz * num_kv_heads * head_dim,
    dtype=torch.int64,
    device=device,
)[:bsz * num_kv_heads * r_channel].view(bsz, num_kv_heads, 1, r_channel) % head_dim

# channel_index: [bsz, num_key_value_heads, 1, r_channel]
def torch_sparq_qk_score(query, key, seq_len, channel_index):
    bsz, q_head, _, head_dim = query.shape
    _, kv_head, _, _ = key.shape
    num_key_value_groups = q_head // kv_head
    r_channel = channel_index.shape[-1]
    query_index = channel_index.expand(-1, -1, num_key_value_groups, -1).reshape(bsz, q_head, 1, r_channel)
    query_partial = torch.gather(query, -1, query_index)
    key_partial = torch.gather(key, -1, channel_index.expand(bsz, kv_head, seq_len, r_channel))
    key_partial = repeat_kv(key_partial, num_key_value_groups)
    topk_score = query_partial @ key_partial.transpose(-1, -2)
    scale = torch.sqrt(
        head_dim
        * query_partial.abs().sum(dim=-1, keepdim=True)
        / query.abs().sum(dim=-1, keepdim=True)
    )
    topk_score = topk_score / scale
    # topk_score = torch.softmax(topk_score.to(torch.float32), dim=-1).to(torch.float16)
    if num_key_value_groups > 1:
        topk_score = topk_score.view(bsz, kv_head, num_key_value_groups, 1, -1).sum(2)
    return topk_score


torch_out = torch_sparq_qk_score(query, key, seq_len, channel_index)
triton_out = sparq_qk_score(query.transpose(1, 2), key.transpose(1, 2), seq_len, channel_index.transpose(1, 2)).reshape(bsz, num_kv_heads, 1, seq_len)

# print(torch_out)
# print(torch_out.shape)
# print(triton_out)
# print(triton_out.shape)

# print(torch_out - triton_out)
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


bench(partial(torch_sparq_qk_score, query, key, seq_len, channel_index))
bench(partial(sparq_qk_score, query.transpose(1, 2), key.transpose(1, 2), seq_len, channel_index.transpose(1, 2)))
