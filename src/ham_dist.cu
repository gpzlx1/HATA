#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "cp_async.cuh"
#include "operator.h"

namespace kvlib {

template <typename T, typename O, int32_t WarpSize, int32_t BLOCK_M,
          bool USE_INT64>
__global__ void hamming_kernel(void* __restrict__ keys_ptr,
                               void* __restrict__ query_ptr,
                               void* __restrict__ output_ptr, int32_t BSZ,
                               int32_t SEQ, int32_t num_kv_head,
                               int32_t num_head, int32_t num_chunk) {
  assert(BLOCK_M == blockDim.y);
  int32_t lane_id = threadIdx.x;
  int32_t warp_id = threadIdx.y;

  int32_t batch_id = blockIdx.x;
  int32_t seq_id = blockIdx.y * blockDim.y + warp_id;

  extern __shared__ int8_t smem_popc[];

  if (seq_id >= SEQ) return;

  int8_t* warp_smem_popc = smem_popc + warp_id * num_head * num_chunk;

  T* q_ptr = (T*)query_ptr + batch_id * 1 * num_head * num_chunk;
  T* k_ptr = (T*)keys_ptr + batch_id * SEQ * num_kv_head * num_chunk +
             seq_id * num_kv_head * num_chunk;
  O* o_ptr = (O*)output_ptr + batch_id * SEQ * num_head + seq_id * num_head;

  int32_t kv_group = num_head / num_kv_head;

#pragma unroll
  for (int32_t i = lane_id; i < num_head * num_chunk; i += WarpSize) {
    int32_t head_id = i / num_chunk;
    int32_t kv_idx = head_id / kv_group;
    T q = q_ptr[i];
    T k = k_ptr[kv_idx * num_chunk + i % num_chunk];
    warp_smem_popc[i] =
        USE_INT64 ? (int8_t)(__popcll(q ^ k)) : (int8_t)(__popc(q ^ k));
  }

  __syncwarp();

  // begin acc
#pragma unroll
  for (int32_t i = lane_id; i < num_head; i += WarpSize) {
    int32_t dist = 0;
#pragma unroll
    for (int32_t j = 0; j < num_chunk; j++) {
      dist += warp_smem_popc[i * num_chunk + j];
    }
    o_ptr[i] = O(dist);
  }
}

torch::Tensor HammingCUDA(torch::Tensor keys, torch::Tensor querys) {
  // shape for keys is (BATCH_SIZE, SEQ, #NUM_K_HEAD, num_chunk) and dtype is
  // int32 shape for querys is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk) and dtype
  // is int32
  int32_t bsz = keys.size(0);
  int32_t seq_len = keys.size(1);
  int32_t num_kv_head = keys.size(2);
  int32_t num_chunk = keys.size(3);

  int32_t num_head = querys.size(2);

  auto device = keys.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor output = torch::empty({bsz, seq_len, num_head}, options);

  constexpr int32_t BLOCK_M = 8;

  // one warp for one key (KV_HEAD, HEAD_DIM) and query (HEAD, HEAD_DIM)
  dim3 blks(32, BLOCK_M);
  dim3 grids(bsz, (seq_len + BLOCK_M - 1) / BLOCK_M);

  if (num_chunk % 2 == 0) {
    num_chunk = num_chunk / 2;
    size_t shm_size = BLOCK_M * num_head * num_chunk *
                      sizeof(int8_t);  // used to store tmp popc results
    hamming_kernel<int64_t, half, 32, BLOCK_M, true>
        <<<grids, blks, shm_size, stream>>>(keys.data_ptr(), querys.data_ptr(),
                                            output.data_ptr(), bsz, seq_len,
                                            num_kv_head, num_head, num_chunk);

  } else {
    size_t shm_size = BLOCK_M * num_head * num_chunk *
                      sizeof(int8_t);  // used to store tmp popc results
    hamming_kernel<int32_t, half, 32, BLOCK_M, false>
        <<<grids, blks, shm_size, stream>>>(keys.data_ptr(), querys.data_ptr(),
                                            output.data_ptr(), bsz, seq_len,
                                            num_kv_head, num_head, num_chunk);
  }

  return output;
}

template <typename T, int32_t WarpSize, int32_t BLOCK_M, bool USE_INT64,
          bool USE_KEY_NORM>
__global__ void hamming_score_kernel(
    void* __restrict__ keys_ptr, void* __restrict__ query_ptr,
    half* __restrict__ output_ptr, int32_t BSZ, int32_t SEQ,
    int32_t num_kv_head, int32_t num_head, int32_t num_chunk,
    half* __restrict__ key_norms, int32_t key_norms_stride0,
    int32_t key_norms_stride1, int32_t key_norms_stride2, int32_t rbit) {
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int batch_id = blockIdx.x;
  const int seq_id = blockIdx.y * BLOCK_M;
  const int max_m_coord = min(SEQ - seq_id, BLOCK_M);

  extern __shared__ int8_t smem[];

  const int elem_in_one_key_code = num_kv_head * num_chunk;
  const int elem_in_one_query_code = num_head * num_chunk;

  int8_t* smem_popc = (int8_t*)(smem);
  half* smem_key_norms = (half*)(smem_popc + BLOCK_M * elem_in_one_query_code);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * elem_in_one_query_code;
  T* k_ptr = (T*)keys_ptr + batch_id * SEQ * elem_in_one_key_code +
             seq_id * elem_in_one_key_code;
  half* k_norm_ptr =
      key_norms + batch_id * key_norms_stride0 + seq_id * key_norms_stride1;

  int32_t kv_group = num_head / num_kv_head;

  // prefetch for key_norms
  constexpr int elem_in_one_pred_load = (128 / 8) / sizeof(half);
  const int total_load_bit_for_key_norms = BLOCK_M * num_kv_head;
  const int num_load_for_key_norms =
      total_load_bit_for_key_norms / elem_in_one_pred_load;
  assert(total_load_bit_for_key_norms % elem_in_one_pred_load == 0);
  assert(num_load_for_key_norms < num_threads);

  flashinfer::cp_async::pred_load<
      128, flashinfer::cp_async::PrefetchMode::kNoPrefetch,
      flashinfer::cp_async::SharedMemFillMode::kNoFill>(
      smem_key_norms + tid * elem_in_one_pred_load,
      k_norm_ptr + tid * elem_in_one_pred_load, tid < num_load_for_key_norms);
  flashinfer::cp_async::commit_group();

  for (int i = tid; i < elem_in_one_query_code * max_m_coord;
       i += num_threads) {
    int m_id = i / elem_in_one_query_code;
    int elem_id = i % elem_in_one_query_code;
    int head_id = elem_id / num_chunk;
    int kv_head_id = head_id / kv_group;

    // if (m_id < max_m_coord) {
    T q = q_ptr[elem_id];
    T k = k_ptr[m_id * elem_in_one_key_code + kv_head_id * num_chunk +
                elem_id % num_chunk];
    smem_popc[i] =
        USE_INT64 ? (int8_t)(__popcll(q ^ k)) : (int8_t)(__popc(q ^ k));
    // }
  }

  flashinfer::cp_async::wait_group<0>();
  __syncthreads();

  half* o_ptr = output_ptr + batch_id * SEQ * num_head; // + seq_id * num_head;

  for (int i = tid; i < num_head * max_m_coord; i += num_threads) {
    int dist = 0;
    int m_id = i / num_head;
    int head_id = i % num_head;
    int kv_head_id = head_id / kv_group;

    for (int j = 0; j < num_chunk; j++) {
      dist += smem_popc[i * num_chunk + j];
    }
    half score;

    if (USE_KEY_NORM) {
      score = ((half)(1.) - half(dist * 2) / half(rbit)) *
              smem_key_norms[m_id * num_kv_head + kv_head_id];
    } else {
      score = (half)(dist);
    }
    // transpose
    // o_ptr[head_id * SEQ + (m_id + seq_id)] = score;

    // no transpose
    o_ptr[i] = score;
  }
}

torch::Tensor HammingScoreCUDA(torch::Tensor key_codes,
                               torch::Tensor query_code,
                               torch::Tensor key_norms, int32_t rbit) {
  // shape for key_codes is (BATCH_SIZE, SEQ, #NUM_K_HEAD, num_chunk) and dtype
  // is int32 shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk) and
  // dtype is int32 shape for key_norms is (BATCH_SIZE, XXX, #NUM_K_HEAD), dtype
  // is float16

  int32_t bsz = key_codes.size(0);
  int32_t seq_len = key_codes.size(1);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);

  int32_t num_head = query_code.size(2);

  int32_t key_norm_stride0 = key_norms.stride(0);
  int32_t key_norm_stride1 = key_norms.stride(1);
  int32_t key_norm_stride2 = key_norms.stride(2);

  // printf("key_norm_stride0: %d, key_norm_stride1: %d, key_norm_stride2:
  // %d\n",
  //        key_norm_stride0, key_norm_stride1, key_norm_stride2);

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor output = torch::empty({bsz, num_head, seq_len}, options);

  constexpr int32_t BLOCK_M = 8;
  dim3 blks(256);
  dim3 grids(bsz, (seq_len + BLOCK_M - 1) / BLOCK_M);

  if (num_chunk % 2 == 0) {
    // convert int32 to int64 to process
    num_chunk = num_chunk / 2;
    size_t shm_size =
        BLOCK_M * num_head * sizeof(half) +  // for cache key_norms
        BLOCK_M * num_head * num_chunk *
            sizeof(int8_t);  // used to store tmp popc results
    hamming_score_kernel<int64_t, 32, BLOCK_M, true, true>
        <<<grids, blks, shm_size, stream>>>(
            key_codes.data_ptr(), query_code.data_ptr(),
            (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_kv_head,
            num_head, num_chunk, (half*)(key_norms.data_ptr<at::Half>()),
            key_norms.stride(0), key_norms.stride(1), key_norms.stride(2),
            rbit);

  } else {
    size_t shm_size =
        BLOCK_M * num_head * sizeof(half) +  // for cache key_norms
        BLOCK_M * num_head * num_chunk *
            sizeof(int8_t);  // used to store tmp popc results
    hamming_score_kernel<int32_t, 32, BLOCK_M, false, true>
        <<<grids, blks, shm_size, stream>>>(
            key_codes.data_ptr(), query_code.data_ptr(),
            (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_kv_head,
            num_head, num_chunk, (half*)(key_norms.data_ptr<at::Half>()),
            key_norms.stride(0), key_norms.stride(1), key_norms.stride(2),
            rbit);
  }

  return output;
}

}  // namespace kvlib