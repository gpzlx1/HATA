#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

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

}  // namespace kvlib