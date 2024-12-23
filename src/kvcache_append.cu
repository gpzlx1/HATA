#include "operator.h"

namespace kvlib {

template <typename T, const int32_t kElemPerThread>
__global__ void kvcache_append_kernel(
    T* __restrict__ dst, T* __restrict__ key, T* __restrict__ value,
    const int32_t insert_pos, const int32_t head_dim,
    const int32_t dst_kv_stride, const int32_t dst_bsz_stride,
    const int32_t dst_seq_stride, const int32_t dst_head_stride,
    const int32_t src_bsz_stride, const int32_t src_head_stride) {
  uint32_t tid = threadIdx.x;
  uint32_t bib = blockIdx.y;
  uint32_t bih = blockIdx.x;

  uint32_t offset = tid * kElemPerThread;
  uint32_t data_type = offset / head_dim;
  uint32_t col_offset = offset % head_dim;

  T* dst_ptr = dst + data_type * dst_kv_stride + bib * dst_bsz_stride +
               insert_pos * dst_seq_stride + bih * dst_head_stride + col_offset;
  T* src_ptr;
  if (data_type == 0) {
    src_ptr = key + bib * src_bsz_stride + bih * src_head_stride + col_offset;
  } else {
    src_ptr = value + bib * src_bsz_stride + bih * src_head_stride + col_offset;
  }

#pragma unroll
  for (uint32_t i = 0; i < kElemPerThread; i += 1) {
    dst_ptr[i] = src_ptr[i];
  }
}

void KVCacheAppend(torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
                   torch::Tensor value_tensor, int32_t insert_pos) {
  int32_t bsz = key_tensor.size(0);
  int32_t num_heads = key_tensor.size(2);
  int32_t head_dim = key_tensor.size(3);

  int32_t dst_kv_stride = kv_cache_tensor.stride(0);
  int32_t dst_bsz_stride = kv_cache_tensor.stride(1);
  int32_t dst_seq_stride = kv_cache_tensor.stride(2);
  int32_t dst_head_stride = kv_cache_tensor.stride(3);

  int32_t src_bsz_stride = key_tensor.stride(0);
  int32_t src_head_stride = key_tensor.stride(2);

  // begin execute kernel
  constexpr int32_t thread_per_block = 32;
  dim3 block(thread_per_block);
  dim3 grid(num_heads, bsz);
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(kv_cache_tensor.device().index());

  kvcache_append_kernel<at::Half, 8><<<grid, block, 0, stream>>>(
      kv_cache_tensor.data_ptr<at::Half>(), key_tensor.data_ptr<at::Half>(),
      value_tensor.data_ptr<at::Half>(), insert_pos, head_dim, dst_kv_stride,
      dst_bsz_stride, dst_seq_stride, dst_head_stride, src_bsz_stride,
      src_head_stride);
}

template <typename T, const int32_t kElemPerThread>
__global__ void kvcache_append_kernel2(
    T* __restrict__ dst, T* __restrict__ src, const int32_t dst_pos,
    const int32_t src_pos, const int32_t head_dim, const int32_t dst_kv_stride,
    const int32_t dst_bsz_stride, const int32_t dst_seq_stride,
    const int32_t dst_head_stride, const int32_t src_kv_stride,
    const int32_t src_bsz_stride, const int32_t src_seq_stride,
    const int32_t src_head_stride) {
  uint32_t tid = threadIdx.x;
  uint32_t bib = blockIdx.y;
  uint32_t bih = blockIdx.x;

  uint32_t offset = tid * kElemPerThread;
  uint32_t data_type = offset / head_dim;
  uint32_t col_offset = offset % head_dim;

  T* dst_ptr = dst + data_type * dst_kv_stride + bib * dst_bsz_stride +
               dst_pos * dst_seq_stride + bih * dst_head_stride + col_offset;
  T* src_ptr = src + data_type * src_kv_stride + bib * src_bsz_stride +
               src_pos * src_seq_stride + bih * src_head_stride + col_offset;

#pragma unroll
  for (uint32_t i = 0; i < kElemPerThread; i += 1) {
    dst_ptr[i] = src_ptr[i];
  }
}

void KVCacheAppend2(torch::Tensor dst_kv_cache_tensor,
                    torch::Tensor src_kv_cache_tensor, int32_t dst_pos,
                    int32_t src_pos) {
  int32_t bsz = dst_kv_cache_tensor.size(1);
  int32_t num_heads = dst_kv_cache_tensor.size(3);
  int32_t head_dim = dst_kv_cache_tensor.size(4);

  int32_t dst_kv_stride = dst_kv_cache_tensor.stride(0);
  int32_t dst_bsz_stride = dst_kv_cache_tensor.stride(1);
  int32_t dst_seq_stride = dst_kv_cache_tensor.stride(2);
  int32_t dst_head_stride = dst_kv_cache_tensor.stride(3);

  int32_t src_kv_stride = src_kv_cache_tensor.stride(0);
  int32_t src_bsz_stride = src_kv_cache_tensor.stride(1);
  int32_t src_seq_stride = src_kv_cache_tensor.stride(2);
  int32_t src_head_stride = src_kv_cache_tensor.stride(3);

  // begin execute kernel
  constexpr int32_t thread_per_block = 32;
  dim3 block(thread_per_block);
  dim3 grid(num_heads, bsz);
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(dst_kv_cache_tensor.device().index());

  kvcache_append_kernel2<at::Half, 8><<<grid, block, 0, stream>>>(
      dst_kv_cache_tensor.data_ptr<at::Half>(),
      src_kv_cache_tensor.data_ptr<at::Half>(), dst_pos, src_pos, head_dim,
      dst_kv_stride, dst_bsz_stride, dst_seq_stride, dst_head_stride,
      src_kv_stride, src_bsz_stride, src_seq_stride, src_head_stride);
}

}  // namespace kvlib