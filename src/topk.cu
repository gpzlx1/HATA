#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <raft/matrix/detail/select_radix.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>

#include "operator.h"

namespace kvlib {

class my_custom_resource : public rmm::mr::device_memory_resource {
  /* implement do_allocate and do_deallocate */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    stream = (cudaStream_t)stream;
    return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bytes,
                                                                  stream);
  }

  void do_deallocate(void* ptr, std::size_t bytes,
                     rmm::cuda_stream_view stream = rmm::cuda_stream_view{}) {
    stream = (cudaStream_t)stream;
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }

  // Get free and available memory for memory resource
  std::pair<std::size_t, std::size_t> do_get_mem_info(
      rmm::cuda_stream_view stream) const noexcept override {
    return std::make_pair(0, 0);
  }

  bool supports_streams() const noexcept override { return true; }

  bool supports_get_mem_info() const noexcept override { return false; }
};

torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest) {
  // note for data, its shape must be [batch_size, num_head, seq_len]
  // may need transpose before this function call
  CHECK(data.device().is_cuda() && data.is_contiguous());
  int32_t batch_size = data.size(0);
  int32_t num_head = data.size(1);
  int32_t seq_len = data.size(2);

  int32_t total_batch_size = batch_size * num_head;

  auto device = data.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor topk_values =
      torch::empty({batch_size, num_head, k},
                   torch::TensorOptions().dtype(data.dtype()).device(device));
  torch::Tensor topk_indices = torch::empty({batch_size, num_head, k}, options);

  my_custom_resource my_mr;

  if (data.dtype() == torch::kFloat32) {
    raft::matrix::detail::select::radix::select_k<float, int32_t, 11, 512>(
        data.data_ptr<float>(), static_cast<int32_t*>(nullptr),
        total_batch_size, seq_len, k, topk_values.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(), !largest, true, stream, &my_mr);
  } else {
    raft::matrix::detail::select::radix::select_k<half, int32_t, 11, 512>(
        (half*)data.data_ptr<at::Half>(), static_cast<int32_t*>(nullptr),
        total_batch_size, seq_len, k, (half*)topk_values.data_ptr<at::Half>(),
        topk_indices.data_ptr<int32_t>(), !largest, true, stream, &my_mr);
  }

  return topk_indices;
}
}  // namespace kvlib