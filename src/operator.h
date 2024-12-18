#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code,
                               torch::Tensor& key_norms, int32_t rbit,
                               int32_t seq_len, bool use_key_norm = false);
torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest);
torch::Tensor EncodeCUDA(torch::Tensor data, torch::Tensor hash_weight);

}  // namespace kvlib