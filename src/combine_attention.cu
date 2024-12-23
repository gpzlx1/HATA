#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

extern "C" {
#include "combine_attention_dim128.h"
}

#include "operator.h"

namespace kvlib {

torch::Tensor combine_attention(torch::Tensor attn1, torch::Tensor lse1,
                                torch::Tensor attn2, torch::Tensor lse2) {
  const int32_t bsz = attn1.size(0);
  const int32_t seq = attn1.size(1);
  const int32_t num_head = attn1.size(2);
  const int32_t head_dim = attn1.size(3);

  CHECK(seq == 1);
  CHECK(head_dim == 128);

  auto device = attn1.device();
  int32_t device_id = device.index();

  torch::Tensor out = torch::empty_like(attn1);

  CUstream stream = (CUstream)c10::cuda::getCurrentCUDAStream(device_id);

  CUdeviceptr attn1_ptr = (CUdeviceptr)attn1.data_ptr<at::Half>();
  CUdeviceptr lse1_ptr = (CUdeviceptr)lse1.data_ptr<float>();

  CUdeviceptr attn2_ptr = (CUdeviceptr)attn2.data_ptr<at::Half>();
  CUdeviceptr lse2_ptr = (CUdeviceptr)lse2.data_ptr<float>();

  CUdeviceptr out_ptr = (CUdeviceptr)out.data_ptr<at::Half>();

  int32_t TOTAL_HEAD = bsz * num_head;

  _combine_attention_dim128_kernel_49e4dd2f_012345(
      stream, attn1_ptr, attn2_ptr, lse1_ptr, lse2_ptr, out_ptr, TOTAL_HEAD);

  return out;
}

}  // namespace kvlib