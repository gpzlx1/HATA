from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [16, 32, 64, 128] for s in ([1, 2, 4]) for w in [4, 8]
]


# @triton.autotune(configs=configs, key=["HEAD_DIM", "GQA_SIZE"])
@triton.jit
def _loki_gqa_qk_score(
    q_ptr,
    k_ptr,
    o_ptr,
    q_b_stride,
    q_h_stride,
    q_s_stride,
    k_b_stride,
    k_h_stride,
    k_s_stride,
    o_b_stride,
    o_h_stride,
    o_s_stride,
    SEQ_LEN,
    GQA_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bid = tl.program_id(0)
    hid_kv = tl.program_id(1)
    sid = tl.program_id(2) * BLOCK_M

    q_offset = bid * q_b_stride + hid_kv * GQA_SIZE * q_h_stride
    k_offset = bid * k_b_stride + hid_kv * k_h_stride
    o_offset = bid * o_b_stride + hid_kv * o_h_stride

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(GQA_SIZE, HEAD_DIM),
        strides=(q_h_stride, 1),
        offsets=(0, 0),
        block_shape=(GQA_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(k_s_stride, 1),
        offsets=(sid, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(SEQ_LEN, 1),
        strides=(o_s_stride, 1),
        offsets=(sid, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    q_data = tl.load(q_block_ptr, boundary_check=(1, 0), padding_option="zero")
    k_data = tl.load(k_block_ptr, boundary_check=(1, 0), padding_option="zero")
    dtype = q_data.dtype
    q_data = tl.cast(q_data, tl.float32)[None, :, :]
    k_data = tl.cast(k_data, tl.float32)[:, None, :]
    score = tl.sum(tl.sum(q_data * k_data, axis=2), axis=1, keep_dims=True)
    score = tl.cast(score, dtype)
    tl.store(o_block_ptr, score, boundary_check=(1, 0))


def loki_qk_score(query, key, seq_len, partial_dim):
    with torch.cuda.device(query.device):
        BSZ, _, NUM_KV_HEADS, _ = key.shape
        NUM_HEADS = query.shape[2]
        gqa_size = NUM_HEADS // NUM_KV_HEADS

        output = torch.zeros((BSZ, NUM_KV_HEADS, seq_len),
                             device=query.device,
                             dtype=query.dtype)

        extra_kern_args = {}
        grid = lambda args: (
            BSZ,
            NUM_KV_HEADS,
            triton.cdiv(seq_len, args["BLOCK_M"]),
        )
        BLOCK_M = 128

        _loki_gqa_qk_score[grid](
            query,
            key,
            output,
            query.stride(0),
            query.stride(2),
            query.stride(1),
            key.stride(0),
            key.stride(2),
            key.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            seq_len,
            gqa_size,
            partial_dim,
            BLOCK_M,
            **extra_kern_args,
        )

        return output
