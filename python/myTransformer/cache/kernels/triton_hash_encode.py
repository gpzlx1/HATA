from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128] for s in ([1, 2, 4]) for w in [4, 8]
]

# configs = [
#     triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
#     for BM in [16]
#     for s in ([1])
#     for w in [1]
# ]


@triton.autotune(configs=configs, key=["HEAD_DIM", "RBIT"])
@triton.jit
def _hash_encode(
    data_ptr,
    hash_weight_ptr,
    packbit_tensor_ptr,
    output_ptr,
    TOTAL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RBIT: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    Data_block_ptr = tl.make_block_ptr(
        base=data_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(TOTAL_HEAD, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load K
    data = tl.load(Data_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weight) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Output_block_ptr, acc, boundary_check=(1, 0))

        # move on
        Weight_ptr = tl.advance(Weight_ptr, (0, BLOCK_N))
        Output_block_ptr = tl.advance(Output_block_ptr, (0, 1))


# @torch.compile(fullgraph=True)
def hash_encode(
        data: torch.Tensor, hash_weight: torch.Tensor,
        packbit_aux_tensor: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.device(data.device):
        assert data.is_contiguous()

        RBIT = hash_weight.shape[1]
        assert RBIT % 32 == 0

        output_dtype = torch.int32
        num_chunk = RBIT // 32
        chunk_size = 32

        if len(data.shape) == 4:
            BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape
            output_shape = (BSZ, SEQ, NUM_HEAD, num_chunk)
            TOTAL_HEAD = BSZ * SEQ * NUM_HEAD
        elif len(data.shape) == 3:
            BSZ, NUM_HEAD, HEAD_DIM = data.shape
            output_shape = (BSZ, NUM_HEAD, num_chunk)
            TOTAL_HEAD = BSZ * NUM_HEAD
        else:
            TOTAL_HEAD, HEAD_DIM = data.shape
            output_shape = (TOTAL_HEAD, num_chunk)

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        assert packbit_aux_tensor.numel() == chunk_size

        output = torch.empty(output_shape,
                             dtype=output_dtype,
                             device=data.device)

        extra_kern_args = {}

        grid = lambda args: (
            triton.cdiv(TOTAL_HEAD, args["BLOCK_M"]),
            1,
            1,
        )
        _hash_encode[grid](
            data,
            hash_weight,
            packbit_aux_tensor,
            output,
            TOTAL_HEAD,
            HEAD_DIM,
            RBIT,
            num_chunk,
            chunk_size,
            BLOCK_N=chunk_size,
            **extra_kern_args,
        )
        return output


@triton.autotune(configs=configs, key=["HEAD_DIM", "RBIT"])
@triton.jit
def _prefill_hash_encode(
    data_ptr,
    hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_norm_output_ptr,
    TOTAL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RBIT: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    Data_block_ptr = tl.make_block_ptr(
        base=data_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Output_block_ptr = tl.make_block_ptr(
        base=key_code_output_ptr,
        shape=(TOTAL_HEAD, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Norm_block_ptr = tl.make_block_ptr(
        base=key_norm_output_ptr,
        shape=(TOTAL_HEAD, 1),
        strides=(1, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load K
    data = tl.load(Data_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    # norm
    norm = tl.sum(data * data, axis=1).reshape(BLOCK_M, 1)
    norm = tl.cast(tl.sqrt(tl.cast(norm, tl.float32)), tl.float16)
    tl.store(Norm_block_ptr, norm, boundary_check=(1, 0))

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weight) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Output_block_ptr, acc, boundary_check=(1, 0))

        # move on
        Weight_ptr = tl.advance(Weight_ptr, (0, BLOCK_N))
        Output_block_ptr = tl.advance(Output_block_ptr, (0, 1))


def prefill_hash_encode(
        data: torch.Tensor, hash_weight: torch.Tensor,
        key_code_output: torch.Tensor, key_norm_output: torch.Tensor,
        packbit_aux_tensor: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # key_code_output: [bsz, seq, num_head, num_chunk]
    # key_norm_output: [bsz, seq, num_head]
    # data: [bsz, seq, num_head, head_dim]

    with torch.cuda.device(data.device):
        assert data.is_contiguous()

        RBIT = hash_weight.shape[1]
        assert RBIT % 32 == 0
        num_chunk = RBIT // 32
        chunk_size = 32

        BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape
        TOTAL_HEAD = BSZ * SEQ * NUM_HEAD

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        assert packbit_aux_tensor.numel() == chunk_size

        extra_kern_args = {}

        grid = lambda args: (
            triton.cdiv(TOTAL_HEAD, args["BLOCK_M"]),
            1,
            1,
        )

        _prefill_hash_encode[grid](
            data,
            hash_weight,
            packbit_aux_tensor,
            key_code_output,
            key_norm_output,
            TOTAL_HEAD,
            HEAD_DIM,
            RBIT,
            num_chunk,
            chunk_size,
            BLOCK_N=chunk_size,
            **extra_kern_args,
        )


@triton.autotune(configs=configs, key=["HEAD_DIM", "RBIT"])
@triton.jit
def _decode_hash_encode(
    key_data_ptr,
    query_data_ptr,
    hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_norm_output_ptr,
    query_code_output_ptr,
    TOTAL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RBIT: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    K_data_block_ptr = tl.make_block_ptr(
        base=key_data_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Q_data_block_ptr = tl.make_block_ptr(
        base=query_data_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )
    K_output_block_ptr = tl.make_block_ptr(
        base=key_code_output_ptr,
        shape=(TOTAL_HEAD, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, start_n),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Q_output_block_ptr = tl.make_block_ptr(
        base=query_code_output_ptr,
        shape=(TOTAL_HEAD, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, start_n),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    K_norm_block_ptr = tl.make_block_ptr(
        base=key_norm_output_ptr,
        shape=(TOTAL_HEAD, 1),
        strides=(1, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    # load Q, K
    k_data = tl.load(K_data_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    # norm
    if start_n == 0:
        norm = tl.sum(k_data * k_data, axis=1).reshape(BLOCK_M, 1)
        norm = tl.cast(tl.sqrt(tl.cast(norm, tl.float32)), tl.float16)
        tl.store(K_norm_block_ptr, norm, boundary_check=(1, 0))

    q_data = tl.load(Q_data_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]

    k_acc = tl.dot(k_data, weight) > 0  # [BLOCK_M, BLOCK_N]
    k_acc = k_acc.to(packbit_tensor.type.element_ty)
    k_acc = k_acc * packbit_tensor
    k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
    tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))

    q_acc = tl.dot(q_data, weight) > 0  # [BLOCK_M, BLOCK_N]
    q_acc = q_acc.to(packbit_tensor.type.element_ty)
    q_acc = q_acc * packbit_tensor
    q_acc = tl.sum(q_acc, axis=1).reshape(BLOCK_M, 1)
    tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))


def decode_hash_encode(
        key_data: torch.Tensor, hash_weight: torch.Tensor,
        key_code_output: torch.Tensor, key_norm_output: torch.Tensor,
        query_data: torch.Tensor, query_code_output: torch.Tensor,
        packbit_aux_tensor: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # code_output: [bsz, seq, num_head, num_chunk]
    # norm_output: [bsz, seq, num_head]
    # data: [bsz, seq, num_head, head_dim]

    with torch.cuda.device(key_data.device):
        assert key_data.is_contiguous()

        RBIT = hash_weight.shape[1]
        assert RBIT % 32 == 0
        NUM_CHUNK = RBIT // 32
        CHUNK_SIZE = 32

        BSZ, SEQ, NUM_HEAD, HEAD_DIM = key_data.shape
        TOTAL_HEAD = BSZ * SEQ * NUM_HEAD

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        assert packbit_aux_tensor.numel() == CHUNK_SIZE

        extra_kern_args = {}

        grid = lambda args: (
            triton.cdiv(TOTAL_HEAD, args["BLOCK_M"]),
            NUM_CHUNK,
            1,
        )

        _decode_hash_encode[grid](
            key_data,
            query_data,
            hash_weight,
            packbit_aux_tensor,
            key_code_output,
            key_norm_output,
            query_code_output,
            TOTAL_HEAD,
            HEAD_DIM,
            RBIT,
            NUM_CHUNK,
            CHUNK_SIZE,
            BLOCK_N=CHUNK_SIZE,
            **extra_kern_args,
        )
