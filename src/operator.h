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
void decode_hash_encode(torch::Tensor key_data, torch::Tensor hash_weights,
                        torch::Tensor key_code_output,
                        torch::Tensor key_norm_output, torch::Tensor query_data,
                        torch::Tensor query_code_output,
                        torch::Tensor packbit_aux_tensor, int32_t cur_seq);

}  // namespace kvlib