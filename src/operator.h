#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code,
                               torch::Tensor& key_norms, int32_t rbit,
                               int32_t seq_len, int32_t sink = 0,
                               int32_t recent = 0, bool use_key_norm = false);
torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest);
void decode_hash_encode(torch::Tensor key_data, torch::Tensor hash_weights,
                        torch::Tensor key_code_output,
                        torch::Tensor key_norm_output, torch::Tensor query_data,
                        torch::Tensor query_code_output,
                        torch::Tensor packbit_aux_tensor, int32_t cur_seq);
void decode_multi_hash_encode(torch::Tensor key_data, torch::Tensor hash_weights,
                        torch::Tensor key_code_output,
                        torch::Tensor key_norm_output, torch::Tensor query_data,
                        torch::Tensor query_code_output,
                        torch::Tensor packbit_aux_tensor, int32_t cur_seq);
torch::Tensor combine_attention(torch::Tensor attn1, torch::Tensor lse1,
                                torch::Tensor attn2, torch::Tensor lse2);
void KVCacheAppend(torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
                   torch::Tensor value_tensor, int32_t insert_pos);
void KVCacheAppend2(torch::Tensor dst_kv_cache_tensor,
                    torch::Tensor src_kv_cache_tensor, int32_t dst_pos,
                    int32_t src_pos);

}  // namespace kvlib