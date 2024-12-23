from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [4, 8, 16, 32] for s in ([1, 2, 4]) for w in [4, 8]
]


# @triton.autotune(configs=configs, key=["HEAD_DIM"])
@triton.jit
def _combine_attention(
    attn_a_ptr,
    attn_b_ptr,
    lse_a_ptr,
    lse_b_ptr,
    output_ptr,
    TOTAL_HEAD,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bid = tl.program_id(0)
    mid = tl.program_id(1)
    Attn_a_block_ptr = tl.make_block_ptr(
        base=attn_a_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(bid, mid * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(1, 0),
    )
    Attn_b_block_ptr = tl.make_block_ptr(
        base=attn_b_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(bid, mid * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(1, 0),
    )
    LSE_a_block_ptr = tl.make_block_ptr(
        base=lse_a_ptr,
        shape=(TOTAL_HEAD, 1),
        strides=(1, 1),
        offsets=(bid, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )
    LSE_b_block_ptr = tl.make_block_ptr(
        base=lse_b_ptr,
        shape=(TOTAL_HEAD, 1),
        strides=(1, 1),
        offsets=(bid, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )
    Output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(bid, mid * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(1, 0),
    )

    attn_a = tl.load(Attn_a_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")
    attn_b = tl.load(Attn_b_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")
    lse_a = tl.load(LSE_a_block_ptr,
                    boundary_check=(1, 0),
                    padding_option="zero")
    lse_b = tl.load(LSE_b_block_ptr,
                    boundary_check=(1, 0),
                    padding_option="zero")

    lse_sum = tl.exp(lse_a) + tl.exp(lse_b)
    attn = (attn_a * tl.exp(lse_a) + attn_b * tl.exp(lse_b)) / lse_sum
    attn = tl.cast(attn, attn_a.dtype)

    tl.store(Output_block_ptr, attn, boundary_check=(1, 0))


def combine_attention(attn_a: torch.Tensor, lse_a: torch.Tensor,
                      attn_b: torch.Tensor, lse_b: torch.Tensor):
    with torch.cuda.device(attn_a.device):
        BSZ, _, NUM_HEADS, HEAD_DIM = attn_a.shape
        TOTAL_HEAD = BSZ * NUM_HEADS

        output = torch.empty_like(attn_a)
        extra_kern_args = {}

        BLOCK_M = 8  # produced by autotuning

        grid = lambda args: (
            TOTAL_HEAD,
            triton.cdiv(HEAD_DIM, args["BLOCK_M"]),
            1,
        )
        _combine_attention[grid](
            attn_a,
            attn_b,
            lse_a,
            lse_b,
            output,
            TOTAL_HEAD,
            HEAD_DIM,
            BLOCK_M,
            **extra_kern_args,
        )
        return output
