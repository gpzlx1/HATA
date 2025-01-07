#pragma once

#include <torch/script.h>

namespace kvlib {

std::vector<torch::Tensor> CPUAttention(torch::Tensor query, torch::Tensor key,
                                        torch::Tensor value, float scale,
                                        int64_t seqlen_k, bool return_lse,
                                        int64_t n_threads);
std::vector<torch::Tensor> CPUSparseAttention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor index, float scale, bool return_lse, int64_t n_threads);

}  // namespace kvlib
