from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [16, 32, 64, 128] for s in ([1, 2, 4]) for w in [4, 8]
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

        def grid(args): return (
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


@triton.jit
def _prefill_hash_encode(
    data_ptr,
    data_stride0,
    hash_weights_ptr,
    packbit_tensor_ptr,
    output_code_output_ptr,
    output_code_stride0,
    output_norm_output_ptr,
    output_norm_stride0,
    TOTAL_LEN,
    BSZ,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)

    DataPtr = data_ptr + batch_id * data_stride0
    OutputCodePtr = output_code_output_ptr + batch_id * output_code_stride0
    OutputNormPtr = output_norm_output_ptr + batch_id * output_norm_stride0

    Data_block_ptr = tl.make_block_ptr(
        base=DataPtr,
        shape=(TOTAL_LEN, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    OutputCode_block_ptr = tl.make_block_ptr(
        base=OutputCodePtr,
        shape=(TOTAL_LEN, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    OutputNorm_block_ptr = tl.make_block_ptr(
        base=OutputNormPtr,
        shape=(TOTAL_LEN, 1),
        strides=(1, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    HashWeights_ptr = tl.make_block_ptr(
        base=hash_weights_ptr,
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
    data_accum = tl.cast(data, tl.float32)
    norm = tl.sum(data_accum * data_accum, axis=1).reshape(BLOCK_M, 1)
    norm = tl.cast(tl.sqrt(norm), tl.float16)
    tl.store(OutputNorm_block_ptr, norm, boundary_check=(1, 0))

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weights = tl.load(HashWeights_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weights) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(OutputCode_block_ptr, acc, boundary_check=(1, 0))

        # move on
        HashWeights_ptr = tl.advance(HashWeights_ptr, (0, BLOCK_N))
        OutputCode_block_ptr = tl.advance(OutputCode_block_ptr, (0, 1))


def prefill_hash_encode(data: torch.Tensor, hash_weights: torch.Tensor,
                        data_code_output: torch.Tensor,
                        data_norm_output: torch.Tensor,
                        packbit_aux_tensor: torch.Tensor) -> None:
    """
    data: [bsz, seq, num_head, head_dim]
    hash_weights: [head_dim, rbit]
    data_code_output: [bsz, xx, num_head, num_chunk] (xx > seq)
    data_norm_output: [bsz, xx, num_head] (xx > seq)
    packbit_aux_tensor: [32]
    """

    with torch.cuda.device(data.device):
        assert data.is_contiguous()

        RBIT = hash_weights.shape[1]
        assert RBIT % 32 == 0

        BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape

        TOTAL_LEN = SEQ * NUM_HEAD

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        BLOCK_M = 128

        def grid(args): return (
            triton.cdiv(TOTAL_LEN, BLOCK_M),
            BSZ,
            1,
        )
        _prefill_hash_encode[grid](
            data,
            data.stride(0),
            hash_weights,
            packbit_aux_tensor,
            data_code_output,
            data_code_output.stride(0),
            data_norm_output,
            data_norm_output.stride(0),
            TOTAL_LEN,
            BSZ,
            RBIT,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            num_stages=2,
            num_warps=4,
        )


@triton.jit
def _prefill_multi_hash_encode(
    data_ptr,
    data_stride0,
    hash_weights_ptr,
    packbit_tensor_ptr,
    output_code_output_ptr,
    output_code_stride0,
    output_norm_output_ptr,
    output_norm_stride0,
    seq_start,
    SEQ,
    BSZ,
    NUM_HEAD: tl.constexpr,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    DataPtr = data_ptr + batch_id * data_stride0 + head_id * HEAD_DIM
    OutputCodePtr = output_code_output_ptr + batch_id * \
        output_code_stride0 + head_id * NUM_CHUNK + seq_start * NUM_HEAD * NUM_CHUNK
    OutputNormPtr = output_norm_output_ptr + \
        batch_id * output_norm_stride0 + head_id * 1 + seq_start * NUM_HEAD
    HashWeightPtr = head_id * (HEAD_DIM * RBIT)

    Data_block_ptr = tl.make_block_ptr(
        base=DataPtr,
        shape=(SEQ, HEAD_DIM),
        strides=(NUM_HEAD * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    OutputCode_block_ptr = tl.make_block_ptr(
        base=OutputCodePtr,
        shape=(SEQ, NUM_CHUNK),
        strides=(NUM_HEAD * NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    OutputNorm_block_ptr = tl.make_block_ptr(
        base=OutputNormPtr,
        shape=(SEQ, 1),
        strides=(NUM_HEAD * 1, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    HashWeights_ptr = tl.make_block_ptr(
        base=hash_weights_ptr + HashWeightPtr,
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
    data_accum = tl.cast(data, tl.float32)
    norm = tl.sum(data_accum * data_accum, axis=1).reshape(BLOCK_M, 1)
    norm = tl.cast(tl.sqrt(norm), tl.float16)
    tl.store(OutputNorm_block_ptr, norm, boundary_check=(1, 0))

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weights = tl.load(HashWeights_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weights) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(OutputCode_block_ptr, acc, boundary_check=(1, 0))

        # move on
        HashWeights_ptr = tl.advance(HashWeights_ptr, (0, BLOCK_N))
        OutputCode_block_ptr = tl.advance(OutputCode_block_ptr, (0, 1))


def prefill_multi_hash_encode(data: torch.Tensor, hash_weights: torch.Tensor,
                              data_code_output: torch.Tensor,
                              data_norm_output: torch.Tensor,
                              packbit_aux_tensor: torch.Tensor, seq_start: int) -> None:
    """
    data: [bsz, seq, num_head, head_dim]
    hash_weights: [head_dim, rbit]
    data_code_output: [bsz, xx, num_head, num_chunk] (xx > seq)
    data_norm_output: [bsz, xx, num_head] (xx > seq)
    packbit_aux_tensor: [32]
    """

    with torch.cuda.device(data.device):
        assert data.is_contiguous()

        RBIT = hash_weights.shape[2]
        assert RBIT % 32 == 0

        BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape

        # TOTAL_LEN = SEQ * NUM_HEAD

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        BLOCK_M = 128

        def grid(args): return (
            triton.cdiv(SEQ, BLOCK_M),
            BSZ,
            NUM_HEAD,
        )
        _prefill_multi_hash_encode[grid](
            data,
            data.stride(0),
            hash_weights,
            packbit_aux_tensor,
            data_code_output,
            data_code_output.stride(0),
            data_norm_output,
            data_norm_output.stride(0),
            seq_start,
            SEQ,
            BSZ,
            NUM_HEAD,
            RBIT,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            num_stages=2,
            num_warps=4,
        )


@triton.jit
def _decode_hash_encode(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    key_norm_output_ptr,
    key_norm_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    CUR_SEQ,
    BSZ,
    MIN_TOTAL_HAED,
    MAX_TOTAL_HEAD,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32
    BLOCK_N: tl.constexpr = CHUNK_SIZE

    SLICE = tl.cdiv(MAX_TOTAL_HEAD, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)
    start_n = tl.program_id(2)

    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]

    if start_m < SLICE:
        # do operator for query
        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0,
            shape=(MAX_TOTAL_HEAD, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0,
            shape=(MAX_TOTAL_HEAD, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, weight) > 0  # [BLOCK_M, BLOCK_N]
        q_acc = q_acc.to(packbit_tensor.type.element_ty)
        q_acc = q_acc * packbit_tensor
        q_acc = tl.sum(q_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    else:
        start_m = start_m - SLICE

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0,
            shape=(MIN_TOTAL_HAED, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            CUR_SEQ * MIN_TOTAL_HAED * NUM_CHUNK,
            shape=(MIN_TOTAL_HAED, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        K_norm_block_ptr = tl.make_block_ptr(
            base=key_norm_output_ptr + batch_id * key_norm_output_stride0 +
            CUR_SEQ * MIN_TOTAL_HAED,
            shape=(MIN_TOTAL_HAED, 1),
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

        k_acc = tl.dot(k_data, weight) > 0  # [BLOCK_M, BLOCK_N]
        k_acc = k_acc.to(packbit_tensor.type.element_ty)
        k_acc = k_acc * packbit_tensor
        k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def decode_hash_encode(key_data: torch.Tensor, hash_weights: torch.Tensor,
                       key_code_output: torch.Tensor,
                       key_norm_output: torch.Tensor, query_data: torch.Tensor,
                       query_code_output: torch.Tensor,
                       packbit_aux_tensor: torch.Tensor, cur_seq: int):

    with torch.cuda.device(key_data.device):
        assert key_data.is_contiguous()

        RBIT = hash_weights.shape[1]
        assert RBIT % 32 == 0

        NUM_CHUNK = RBIT // 32

        BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
        NUM_HEAD = query_data.shape[2]

        assert SEQ == 1
        assert HEAD_DIM in {128, 256}

        MIN_TOTAL_HEAD = NUM_KV_HEAD
        MAX_TOTAL_HEAD = NUM_HEAD

        BLOCK_M = 16

        def grid(args): return (
            triton.cdiv(MAX_TOTAL_HEAD, BLOCK_M) + triton.cdiv(
                MIN_TOTAL_HEAD, BLOCK_M),
            BSZ,
            NUM_CHUNK,
        )

        _decode_hash_encode[grid](
            key_data,
            key_data.stride(0),
            query_data,
            query_data.stride(0),
            hash_weights,
            packbit_aux_tensor,
            key_code_output,
            key_code_output.stride(0),
            key_norm_output,
            key_norm_output.stride(0),
            query_code_output,
            query_code_output.stride(0),
            cur_seq,
            BSZ,
            MIN_TOTAL_HEAD,
            MAX_TOTAL_HEAD,
            RBIT,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=1,
        )

# @triton.autotune(configs=configs, key=["HEAD_DIM", "RBIT"])


@triton.jit
def _decode_multi_hash_encode(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    key_norm_output_ptr,
    key_norm_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    CUR_SEQ,
    BSZ,
    KV_HEAD,
    Q_HEAD,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32
    BLOCK_N: tl.constexpr = CHUNK_SIZE

    KV_GROUP = Q_HEAD // KV_HEAD

    SLICE = tl.cdiv(KV_GROUP, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr + head_id * HEAD_DIM * RBIT,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]

    if start_m < SLICE:
        # do operator for query
        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHUNK,
            shape=(KV_GROUP, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, weight) > 0  # [BLOCK_M, BLOCK_N]
        q_acc = q_acc.to(packbit_tensor.type.element_ty)
        q_acc = q_acc * packbit_tensor
        q_acc = tl.sum(q_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    else:
        start_m = start_m - SLICE

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0 +
            head_id * HEAD_DIM,
            shape=(1, HEAD_DIM),
            strides=(KV_HEAD * HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            CUR_SEQ * KV_HEAD * NUM_CHUNK + head_id * NUM_CHUNK,
            shape=(1, NUM_CHUNK),
            strides=(KV_HEAD * NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        K_norm_block_ptr = tl.make_block_ptr(
            base=key_norm_output_ptr + batch_id * key_norm_output_stride0 +
            CUR_SEQ * KV_HEAD + head_id * 1,
            shape=(1, 1),
            strides=(KV_HEAD * 1, 1),
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

        k_acc = tl.dot(k_data, weight) > 0  # [BLOCK_M, BLOCK_N]
        k_acc = k_acc.to(packbit_tensor.type.element_ty)
        k_acc = k_acc * packbit_tensor
        k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def decode_multi_hash_encode(
        key_data: torch.Tensor, hash_weights: torch.Tensor,
        key_code_output: torch.Tensor, key_norm_output: torch.Tensor,
        query_data: torch.Tensor, query_code_output: torch.Tensor,
        packbit_aux_tensor: torch.Tensor, cur_seq: int):

    with torch.cuda.device(key_data.device):
        assert key_data.is_contiguous()

        RBIT = hash_weights.shape[2]
        assert RBIT % 32 == 0

        NUM_CHUNK = RBIT // 32

        BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
        NUM_HEAD = query_data.shape[2]

        assert SEQ == 1
        # assert HEAD_DIM in {128, 256}

        # MIN_TOTAL_HEAD = NUM_KV_HEAD
        # MAX_TOTAL_HEAD = NUM_HEAD
        KV_GROUP = NUM_HEAD // NUM_KV_HEAD

        BLOCK_M = 16

        def grid(args): return (
            triton.cdiv(KV_GROUP * SEQ, BLOCK_M) + triton.cdiv(SEQ, BLOCK_M),
            BSZ * NUM_KV_HEAD,
            NUM_CHUNK,
        )

        _decode_multi_hash_encode[grid](
            key_data,
            key_data.stride(0),
            query_data,
            query_data.stride(0),
            hash_weights,
            packbit_aux_tensor,
            key_code_output,
            key_code_output.stride(0),
            key_norm_output,
            key_norm_output.stride(0),
            query_code_output,
            query_code_output.stride(0),
            cur_seq,
            BSZ,
            NUM_KV_HEAD,
            NUM_HEAD,
            RBIT,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=1,
        )
