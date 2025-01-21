import torch
from functools import partial
from myTransformer.cache.kernels.triton_qk_score import sparq_qk_score, sparq_qk_score_v2
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

bsz = 1
num_heads = 32
num_kv_heads = 32
seq_len = 100000
head_dim = 128
r_channel = 32

device = "cuda:7"
torch.cuda.set_device(device)
dtype = torch.float16

query = torch.randn(
    (bsz, 1, num_heads, head_dim),
    dtype=torch.float16,
    device=device,
)
key = torch.randn(
    (bsz, seq_len, num_kv_heads, head_dim),
    dtype=torch.float16,
    device=device,
)

kv_group = num_heads // num_kv_heads
# compute index
abs_query = query.abs()
reduce_abs_query = abs_query.view(bsz, num_kv_heads, kv_group, 1,
                                  head_dim).sum(2)
channel_index = torch.topk(
    reduce_abs_query, r_channel,
    dim=-1).indices  # shape [bsz, num_kv_heads, 1, r_channel]


# channel_index: [bsz, num_key_value_heads, 1, r_channel]
def torch_sparq_qk_score(query, key, seq_len, channel_index):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)

    bsz, q_head, _, head_dim = query.shape
    _, kv_head, _, _ = key.shape
    num_key_value_groups = q_head // kv_head

    # expand channel_index
    r_channel = channel_index.shape[-1]
    channel_index_expand = channel_index.expand(-1, -1, num_key_value_groups,
                                                -1).reshape(
                                                    bsz, q_head, 1, r_channel)
    query_partial = torch.gather(query, -1, channel_index_expand)
    key_partial = torch.gather(
        key, -1, channel_index.expand(bsz, kv_head, seq_len, r_channel))
    key_partial = repeat_kv(key_partial, num_key_value_groups)

    topk_score = query_partial @ key_partial.transpose(-1, -2)
    scale = torch.sqrt(head_dim *
                       query_partial.abs().sum(dim=-1, keepdim=True) /
                       query.abs().sum(dim=-1, keepdim=True))
    topk_score = topk_score / scale
    topk_score = torch.softmax(topk_score.to(torch.float32),
                               dim=-1).to(torch.float16)

    topk_score = topk_score.view(bsz, kv_head, num_key_value_groups, -1).sum(2)
    return topk_score


def triton_sparq_qk_score(query, key, seq_len, channel_index):
    channel_index_expand = channel_index.expand(-1, -1, kv_group, -1).reshape(
        bsz, 1, num_heads, r_channel)

    partial_query = torch.gather(query, -1, index=channel_index_expand)
    scale = torch.sqrt(head_dim *
                       partial_query.abs().sum(dim=-1, keepdim=True) /
                       query.abs().sum(dim=-1, keepdim=True))

    # compute score
    score = sparq_qk_score_v2(partial_query, key, scale, seq_len,
                              channel_index)
    score = torch.softmax(score.to(torch.float32), dim=-1).to(torch.float16)
    score = score.view(bsz, num_kv_heads, kv_group, seq_len)
    score = torch.sum(score, dim=2)
    return score


# def compute_sparq_metadata(query, r_channel):
#     abs_query = query.abs()
#     reduce_abs_query = abs_query.view(bsz, num_kv_heads, kv_group, 1,
#                                       head_dim).sum(2)

#     # shape [bsz, num_kv_heads, 1, r_channel]
#     channel_index = torch.topk(reduce_abs_query, r_channel, dim=-1).indices
#     channel_index_expand = channel_index.expand(-1, -1, kv_group, -1).reshape(
#         bsz, 1, num_heads, r_channel)
#     partial_query = torch.gather(query, -1, index=channel_index_expand)

#     scale = torch.sqrt(head_dim *
#                        partial_query.abs().sum(dim=-1, keepdim=True) /
#                        query.abs().sum(dim=-1, keepdim=True))

#     # compute score
#     score = sparq_qk_score_v2(partial_query, key, scale, seq_len,
#                               channel_index)
#     # score = torch.softmax(score, dim=-1)
#     score = score.view(bsz, num_kv_heads, kv_group, seq_len)
#     score = torch.sum(score, dim=2)
#     return score

torch_out = torch_sparq_qk_score(query, key, seq_len, channel_index)
triton_out = triton_sparq_qk_score(query, key, seq_len, channel_index)

print(torch_out)
print(torch_out.shape)
print(triton_out)
print(triton_out.shape)

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
bench(partial(triton_sparq_qk_score, query, key, seq_len, channel_index))
