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
  CpuAttention(size_t mem_size, int n_dims, const std::vector<int64_t>& ne);
  ~CpuAttention();

  void FillKeyValye(torch::Tensor keys, torch::Tensor values);

  void AppendKeyValue(torch::Tensor keys, torch::Tensor values);

  //   struct ggml_tensor* ConvertFromTorchTensor(torch::Tensor t) {}

  void Attention(torch::Tensor query, torch::Tensor result);
  void SparseAttention(torch::Tensor query, torch::Tensor result,
                       torch::Tensor index);

  void SparseAttentionWithMeta(torch::Tensor query, torch::Tensor result,
                               torch::Tensor index);

 private:
  struct ggml_context* ctx0;         // Holds the context
  struct ggml_tensor* query_buffer;  // Holds the query buffer
  struct ggml_tensor* key_buffer;
  struct ggml_tensor* value_buffer;
  int n_dims;               // Member variable for n_dims
  std::vector<int64_t> ne;  // Member variable for ne
};

}  // namespace kvlib