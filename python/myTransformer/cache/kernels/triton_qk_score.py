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
def _topk_gqa_qk_score(
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


def topk_qk_score(query, key, seq_len):
    with torch.cuda.device(query.device):
        BSZ, _, NUM_KV_HEADS, HEAD_DIM = key.shape
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
        BLOCK_M = 32

        _topk_gqa_qk_score[grid](
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
            HEAD_DIM,
            BLOCK_M,
            **extra_kern_args,
        )

        return output


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

# @triton.autotune(configs=configs, key=["HEAD_DIM", "GQA_SIZE"])
@triton.jit
def _sparq_gqa_qk_score(
    q_ptr,
    k_ptr,
    i_ptr,
    o_ptr,
    q_b_stride,
    q_h_stride,
    q_s_stride,
    k_b_stride,
    k_h_stride,
    k_s_stride,
    i_b_stride,
    i_h_stride,
    i_s_stride,
    o_b_stride,
    o_h_stride,
    o_s_stride,
    SEQ_LEN,
    GQA_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    R_CHANNEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid_kv = tl.program_id(1)
    sid = tl.program_id(2) * BLOCK_M

    q_offset = bid * q_b_stride + hid_kv * GQA_SIZE * q_h_stride
    k_offset = bid * k_b_stride + hid_kv * k_h_stride + sid * k_s_stride
    o_offset = bid * o_b_stride + hid_kv * o_h_stride + sid * o_s_stride
    i_offset = bid * i_b_stride + hid_kv * i_h_stride

    channel_offset = tl.arange(0, R_CHANNEL)
    full_channel_offset = tl.arange(0, HEAD_DIM)
    channel_index = tl.load(i_ptr + i_offset + channel_offset) # [R_CHANNEL]

    gqa_offset = tl.arange(0, GQA_SIZE)
    q_load_offset = q_offset + gqa_offset[:, None] * q_h_stride + channel_index
    q_full_load_offset = q_offset + gqa_offset[:, None] * q_h_stride + full_channel_offset

    block_range = tl.arange(0, BLOCK_M)
    k_load_offset = k_offset + block_range[:, None] * k_s_stride + channel_index

    q_data = tl.load(q_ptr + q_load_offset) # [GQA, R_CHANNEL]
    q_full_data = tl.load(q_ptr + q_full_load_offset) # [GQA, HEAD_DIM]
    k_data = tl.load(k_ptr + k_load_offset, (sid + block_range < SEQ_LEN)[:, None]) # [BLOCK_M, R_CHANNEL]
    dtype = q_data.dtype

    o_store_offset = o_offset + block_range[None, :] * o_s_stride

    q_data = tl.cast(q_data, tl.float32) # [GQA, R_CHANNEL]
    q_full_data = tl.cast(q_full_data, tl.float32)
    scale = tl.sum(tl.abs(q_data), axis=1) / tl.sum(tl.abs(q_full_data), axis=1)
    scale = tl.sqrt(HEAD_DIM * scale)
    q_data = q_data / scale[:, None]
    k_data = tl.cast(k_data, tl.float32) # [BLOCK_M, R_CHANNEL]
    score = tl.sum(tl.sum(q_data[:, None, :] * k_data[None, :, :], axis=2), axis=0, keep_dims=True)
    score = tl.cast(score, dtype)
    tl.store(o_ptr + o_store_offset, score, (sid + block_range < SEQ_LEN)[None, :])


def sparq_qk_score(query, key, seq_len, channel_index):
    with torch.cuda.device(query.device):
        BSZ, _, NUM_KV_HEADS, HEAD_DIM = key.shape
        _, _, _, R_CHANNEL = channel_index.shape
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

        _sparq_gqa_qk_score[grid](
            query,
            key,
            channel_index,
            output,
            query.stride(0),
            query.stride(2),
            query.stride(1),
            key.stride(0),
            key.stride(2),
            key.stride(1),
            channel_index.stride(0),
            channel_index.stride(2),
            channel_index.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            seq_len,
            gqa_size,
            BLOCK_M,
            R_CHANNEL,
            HEAD_DIM,
            **extra_kern_args,
        )

        return output