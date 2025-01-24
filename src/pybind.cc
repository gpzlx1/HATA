#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

#include "cpu-attn/cpu_attn.h"
#include "cuda-attn/flash_api.h"
#include "operator.h"

namespace py = pybind11;

torch::Tensor create_tensor(std::vector<int32_t> size, int dtype) {
  // 计算张量的总元素数量
  int64_t num_elements = 1;
  for (auto dim : size) {
    num_elements *= dim;
  }

  if (dtype == 16) {
    // void* buf = aligned_alloc(64, num_elements * sizeof(half));
    void* buf;
    cudaMallocHost(&buf, num_elements * sizeof(half));

    auto tensor = torch::from_blob(buf, {num_elements}, torch::kFloat16);

    // 返回张量
    return tensor;  // 克隆以使张量持有自己的数据
  } else {
    // void* buf = aligned_alloc(64, num_elements * sizeof(float));
    void* buf;
    cudaMallocHost(&buf, num_elements * sizeof(float));

    auto tensor = torch::from_blob(buf, {num_elements}, torch::kFloat32);

    // 返回张量
    return tensor;  // 克隆以使张量持有自己的数据
  }
}

PYBIND11_MODULE(KVLib, m) {
  m.def("hamming_score", &kvlib::HammingScoreCUDA, py::arg("key_code"),
        py::arg("query_code"), py::arg("key_norm"), py::arg("rbit"),
        py::arg("seq_len"), py::arg("sink") = 0, py::arg("recent") = 0,
        py::arg("use_key_norm") = false)
      .def("batch_topk", &kvlib::TopkCUDA)
      .def("decode_hash_encode", &kvlib::decode_hash_encode)
      .def("decode_multi_hash_encode", &kvlib::decode_multi_hash_encode)
      .def("flash_index_decode", &kvlib::mha_index_decode_fwd)
      .def("flash_decode", &kvlib::mha_decode_fwd)
      .def("combine_attention", &kvlib::combine_attention)
      .def("kvcache_append", &kvlib::KVCacheAppend)
      .def("kvcache_append2", &kvlib::KVCacheAppend2)
      .def("create_tensor", &create_tensor)
      .def("cpu_attn", &kvlib::CPUAttention)
      .def("cpu_sparse_attn", &kvlib::CPUSparseAttention);
}