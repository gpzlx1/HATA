#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "cp_async.cuh"
#include "operator.h"

namespace kvlib {

template <typename T, int32_t WarpSize, int NumHead, int NumKVHead,
          int32_t BLOCK_M, bool USE_INT64>
__global__ void hamming_score_kernel(
    void* __restrict__ keys_ptr, void* __restrict__ query_ptr,
    half* __restrict__ output_ptr, int32_t BSZ, int32_t SEQ, int32_t num_chunk,
    half* __restrict__ key_norms, int32_t key_code_stride0,
    int32_t key_code_stride1, int32_t key_code_stride2,
    int32_t key_norms_stride0, int32_t key_norms_stride1,
    int32_t key_norms_stride2, int32_t rbit, bool use_key_norm) {
  constexpr int num_kv_head = NumKVHead;
  constexpr int num_head = NumHead;
  const int tid = threadIdx.x;
  const int lane_id = threadIdx.x % WarpSize;
  const int warp_id = threadIdx.x / WarpSize;  // in [0, BLOCK_M]

  const int batch_id = blockIdx.x;
  const int m_start = blockIdx.y * BLOCK_M;
  const int seq_id = m_start + warp_id;
  const int max_m_coord = min(SEQ - m_start, BLOCK_M);

  extern __shared__ int8_t smem[];

  const int elem_in_one_query_code = num_head * num_chunk;

  int8_t* warp_smem_popc = (int8_t*)(smem + warp_id * num_head * num_chunk);
  half* smem_score = (half*)(smem + BLOCK_M * num_head * num_chunk);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * elem_in_one_query_code;
  T* k_ptr =
      (T*)keys_ptr + batch_id * key_code_stride0 + seq_id * key_code_stride1;
  half* k_norm_ptr =
      key_norms + batch_id * key_norms_stride0 + seq_id * key_norms_stride1;

  const int32_t kv_group = num_head / num_kv_head;

  if (seq_id < SEQ) {
#pragma unroll
    for (int i = lane_id; i < elem_in_one_query_code; i += warpSize) {
      int head_id = i / num_chunk;
      int kv_head_id = head_id / kv_group;

      T q = q_ptr[i];
      T k = k_ptr[kv_head_id * key_code_stride2 + i % num_chunk];
      warp_smem_popc[i] =
          USE_INT64 ? (int8_t)(__popcll(q ^ k)) : (int8_t)(__popc(q ^ k));
    }
    __syncwarp();

    // transpose
    // half* o_ptr = output_ptr + batch_id * SEQ * num_head + seq_id;

    // no transpose
    // half* o_ptr = output_ptr + batch_id * SEQ * num_head + seq_id * num_head;

#pragma unroll
    for (int i = lane_id; i < num_head; i += warpSize) {
      int dist = 0;
#pragma unroll
      for (int j = 0; j < num_chunk; j++) {
        dist += warp_smem_popc[i * num_chunk + j];
      }
      half score;
      int kv_head_id = i / kv_group;

      if (use_key_norm) {
        score =
            ((half)(1.) - half(dist * 2) / half(rbit)) * k_norm_ptr[kv_head_id];
      } else {
        score = (half)(dist);
      }
      // transpose
      // o_ptr[i * SEQ] = score;
      // save to smem
      smem_score[i * BLOCK_M + warp_id] = score;

      // no transpose
      // o_ptr[i] = score;
    }
  }

  __syncthreads();
  half* o_ptr = output_ptr + batch_id * SEQ * num_kv_head + m_start;
  // write to global memory

  if (kv_group > 1) {
#pragma unroll
    for (int i = tid; i < BLOCK_M * num_kv_head; i += blockDim.x) {
      int kv_head_id = i / BLOCK_M;
      int m_id = i % BLOCK_M;
      if (m_id < max_m_coord) {
        half* _o_ptr = o_ptr + kv_head_id * SEQ + m_id;
        float sum = 0.0;
        int head_id = kv_head_id * kv_group;
        for (int h = 0; h < kv_group; h += 1) {
          sum += (float)smem_score[(head_id + h) * BLOCK_M + m_id];
        }
        *_o_ptr = half(sum);
      }
    }
  } else {
#pragma unroll
    for (int i = tid; i < BLOCK_M * num_head; i += blockDim.x) {
      int head_id = i / BLOCK_M;
      int m_id = i % BLOCK_M;
      if (m_id < max_m_coord) {
        half* _o_ptr = o_ptr + head_id * SEQ + m_id;
        *_o_ptr = smem_score[i];
      }
    }
  }
}

torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code,
                               torch::Tensor& key_norms, int32_t rbit,
                               int32_t seq_len, bool use_key_norm) {
  // shape for key_codes is (BATCH_SIZE, SEQ, #NUM_K_HEAD, num_chunk) and dtype
  // is int32 shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk) and
  // dtype is int32 shape for key_norms is (BATCH_SIZE, XXX, #NUM_K_HEAD), dtype
  // is float16

  int32_t bsz = key_codes.size(0);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);

  int32_t num_head = query_code.size(2);

  int32_t key_norm_stride0 = key_norms.stride(0);
  int32_t key_norm_stride1 = key_norms.stride(1);
  int32_t key_norm_stride2 = key_norms.stride(2);

  int32_t key_code_stride0 = key_codes.stride(0);
  int32_t key_code_stride1 = key_codes.stride(1);
  int32_t key_code_stride2 = key_codes.stride(2);

  CHECK(num_head == 32);
  CHECK(num_kv_head == 32 || num_kv_head == 8);

  // printf("key_norm_stride0: %d, key_norm_stride1: %d, key_norm_stride2: %
  // d\n",
  //        key_norm_stride0, key_norm_stride1, key_norm_stride2);
  // printf("key_code_stride0: %d, key_code_stride1: %d, key_code_stride2: %
  // d\n",
  //        key_code_stride0, key_code_stride1, key_code_stride2);

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor output = torch::empty({bsz, num_kv_head, seq_len}, options);

  constexpr int32_t BLOCK_M = 8;
  dim3 blks(32 * BLOCK_M);
  dim3 grids(bsz, (seq_len + BLOCK_M - 1) / BLOCK_M);

  if (num_chunk % 2 == 0) {
    // convert int32 to int64 to process
    num_chunk = num_chunk / 2;
    key_code_stride0 = key_code_stride0 / 2;
    key_code_stride1 = key_code_stride1 / 2;
    key_code_stride2 = key_code_stride2 / 2;
    size_t shm_size = BLOCK_M * num_head * sizeof(half) +  // for score
                      BLOCK_M * num_head * num_chunk *
                          sizeof(int8_t);  // used to store tmp popc results
    if (num_kv_head == num_head) {
      hamming_score_kernel<int64_t, 32, 32, 32, BLOCK_M, true>
          <<<grids, blks, shm_size, stream>>>(
              key_codes.data_ptr(), query_code.data_ptr(),
              (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_chunk,
              (half*)(key_norms.data_ptr<at::Half>()), key_code_stride0,
              key_code_stride1, key_code_stride2, key_norm_stride0,
              key_norm_stride1, key_norm_stride2, rbit, use_key_norm);
    } else {
      hamming_score_kernel<int64_t, 32, 32, 8, BLOCK_M, true>
          <<<grids, blks, shm_size, stream>>>(
              key_codes.data_ptr(), query_code.data_ptr(),
              (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_chunk,
              (half*)(key_norms.data_ptr<at::Half>()), key_code_stride0,
              key_code_stride1, key_code_stride2, key_norm_stride0,
              key_norm_stride1, key_norm_stride2, rbit, use_key_norm);
    }

  } else {
    size_t shm_size = BLOCK_M * num_head * sizeof(half) +  // for score
                      BLOCK_M * num_head * num_chunk *
                          sizeof(int8_t);  // used to store tmp popc results
    if (num_kv_head == num_head) {
      hamming_score_kernel<int32_t, 32, 32, 32, BLOCK_M, false>
          <<<grids, blks, shm_size, stream>>>(
              key_codes.data_ptr(), query_code.data_ptr(),
              (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_chunk,
              (half*)(key_norms.data_ptr<at::Half>()), key_code_stride0,
              key_code_stride1, key_code_stride2, key_norm_stride0,
              key_norm_stride1, key_norm_stride2, rbit, use_key_norm);
    } else {
      hamming_score_kernel<int32_t, 32, 32, 8, BLOCK_M, false>
          <<<grids, blks, shm_size, stream>>>(
              key_codes.data_ptr(), query_code.data_ptr(),
              (half*)(output.data_ptr<at::Half>()), bsz, seq_len, num_chunk,
              (half*)(key_norms.data_ptr<at::Half>()), key_code_stride0,
              key_code_stride1, key_code_stride2, key_norm_stride0,
              key_norm_stride1, key_norm_stride2, rbit, use_key_norm);
    }
  }

  return output;
}

}  // namespace kvlib