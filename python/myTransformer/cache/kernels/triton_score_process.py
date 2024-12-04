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
    BSZ: tl.constexpr,
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
    BSZ_SEQ = BSZ * SEQ_LEN
    BSZ_HEAD = BSZ * NUM_HEAD
    src_offset = start_m * SEQ_LEN + start_n
    dst_offset = start_m * NUM_HEAD + start_k

    Data_block_ptr = tl.make_block_ptr(
        base=data_ptr,
        shape=(BSZ_SEQ, NUM_HEAD),
        strides=(NUM_HEAD, 1),
        offsets=(src_offset, start_k),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    Norm_block_ptr = tl.make_block_ptr(
        base=norm_ptr,
        shape=(BSZ_SEQ, NUM_K_HEAD),
        strides=(NUM_HEAD, 1),
        offsets=(src_offset, start_k_gqa),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    Out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BSZ_HEAD, SEQ_LEN),
        strides=(SEQ_LEN, 1),
        offsets=(dst_offset, start_n),
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
        assert key_norm.is_contiguous()

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
            BSZ,
            SEQ_LEN,
            NUM_HEAD,
            NUM_K_HEAD,
            GQA_SIZE,
            rbit,
            BLOCK_N=128,
        )

        return out
