from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BN in [32, 64, 128] for s in ([1, 2, 4]) for w in [4, 8]
]


# @triton.autotune(configs=configs, key=[])
@triton.jit
def _score_process(
    data_ptr,
    norm_ptr,
    out_ptr,
    data_b_stride,
    norm_b_stride,
    out_b_stride,
    SEQ_LEN: tl.constexpr,
    NUM_HEAD: tl.constexpr,
    NUM_K_HEAD: tl.constexpr,
    GQA_SIZE: tl.constexpr,
    RBIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_N
    start_k = tl.program_id(2)
    start_k_gqa = start_k // GQA_SIZE
    data_offset = start_m * data_b_stride
    norm_offset = start_m * norm_b_stride
    out_offset = start_m * out_b_stride

    Data_block_ptr = tl.make_block_ptr(
        base=data_ptr + data_offset,
        shape=(SEQ_LEN, NUM_HEAD),
        strides=(NUM_HEAD, 1),
        offsets=(start_n, start_k),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    Norm_block_ptr = tl.make_block_ptr(
        base=norm_ptr + norm_offset,
        shape=(SEQ_LEN, NUM_K_HEAD),
        strides=(NUM_HEAD, 1),
        offsets=(start_n, start_k_gqa),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    Out_block_ptr = tl.make_block_ptr(
        base=out_ptr + out_offset,
        shape=(NUM_HEAD, SEQ_LEN),
        strides=(SEQ_LEN, 1),
        offsets=(start_k, start_n),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    data = tl.load(Data_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_N, 1]
    norm = tl.load(Norm_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_N, 1]
    score = 1 - 2 * data / RBIT
    score = tl.cast(score, norm.dtype)
    score = score * norm

    tl.store(Out_block_ptr, score.trans(), boundary_check=(1, 0))


def hash_score_process(hamming_dist: torch.Tensor, key_norm: torch.Tensor,
                       rbit: int):
    with torch.cuda.device(hamming_dist.device):
        assert hamming_dist.is_contiguous()

        BSZ, SEQ_LEN, NUM_HEAD = hamming_dist.shape
        NUM_K_HEAD = key_norm.shape[2]
        GQA_SIZE = NUM_HEAD // NUM_K_HEAD

        out = torch.empty((BSZ, NUM_HEAD, SEQ_LEN),
                          device=hamming_dist.device,
                          dtype=key_norm.dtype)

        grid = lambda args: (
            BSZ,
            triton.cdiv(SEQ_LEN, args["BLOCK_N"]),
            NUM_HEAD,
        )

        extra_kern_args = {}

        _score_process[grid](
            hamming_dist,
            key_norm,
            out,
            hamming_dist.stride(0),
            key_norm.stride(0),
            out.stride(0),
            SEQ_LEN,
            NUM_HEAD,
            NUM_K_HEAD,
            GQA_SIZE,
            rbit,
            BLOCK_N=128,
            **extra_kern_args,
        )

        return out
