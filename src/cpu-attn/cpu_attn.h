#pragma once

#include <cuda_fp16.h>
#include <torch/script.h>
#include <cmath>
#include "ggml-cpu.h"
#include "ggml.h"

namespace kvlib {

class CpuAttention {
 public:
  void MyGGML_print_tensor(const struct ggml_tensor* tensor);
  void MyGGML_print_tensor_data(const struct ggml_tensor* tensor);
  void MyGGML_print_tensor_data_float(const struct ggml_tensor* tensor);

  // CpuAttention(size_t mem_size, int n_dims, const int64_t* ne);
  CpuAttention(size_t mem_size, int num_threads);
  ~CpuAttention();

  torch::Tensor Attention(torch::Tensor query, torch::Tensor key,
                          torch::Tensor value, float scale);
  torch::Tensor SparseAttention(torch::Tensor query, torch::Tensor key,
                                torch::Tensor value, torch::Tensor index,
                                float scale);
  std::vector<torch::Tensor> SparseAttentionWithMeta(torch::Tensor query,
                                                     torch::Tensor key,
                                                     torch::Tensor value,
                                                     torch::Tensor index,
                                                     float scale);

 private:
  struct ggml_context* ggml_ctx;  // Holds the context
  int num_threads;
};

}  // namespace kvlib