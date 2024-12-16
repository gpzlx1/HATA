#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor AddCUDA(torch::Tensor a, torch::Tensor b, torch::Tensor c);
torch::Tensor HammingCUDA(torch::Tensor keys, torch::Tensor querys);
torch::Tensor HammingScoreCUDA(torch::Tensor key_codes,
                               torch::Tensor query_code,
                               torch::Tensor key_norms, int32_t rbit);
torch::Tensor TopkCUDA(torch::Tensor data, int32_t k, bool largest);
torch::Tensor EncodeCUDA(torch::Tensor data, torch::Tensor hash_weight);

}  // namespace kvlib