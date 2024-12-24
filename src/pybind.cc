#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cpu-attn/cpu_attn.h"
#include "cuda-attn/flash_api.h"
#include "operator.h"

namespace py = pybind11;

PYBIND11_MODULE(KVLib, m) {
  m.def("hamming_score", &kvlib::HammingScoreCUDA, py::arg("key_code"),
        py::arg("query_code"), py::arg("key_norm"), py::arg("rbit"),
        py::arg("seq_len"), py::arg("use_key_norm") = false)
      .def("batch_topk", &kvlib::TopkCUDA)
      .def("decode_hash_encode", &kvlib::decode_hash_encode)
      .def("flash_index_decode", &kvlib::mha_index_decode_fwd)
      .def("flash_decode", &kvlib::mha_decode_fwd)
      .def("combine_attention", &kvlib::combine_attention)
      .def("kvcache_append", &kvlib::KVCacheAppend)
      .def("kvcache_append2", &kvlib::KVCacheAppend2);

  py::class_<kvlib::CpuAttention>(m, "CpuAttention")
      .def(py::init<size_t, int>(), py::arg("mem_size"), py::arg("num_threads"),
           "Initialize CpuAttention with memory size, dimensions, and shape")
      .def("Attention", &kvlib::CpuAttention::Attention,
           "Compute Attention in CPU")
      .def("SparseAttention", &kvlib::CpuAttention::SparseAttention,
           "Compute Sparse Attention in CPU")
      .def("SparseAttentionWithMeta",
           &kvlib::CpuAttention::SparseAttentionWithMeta,
           "Compute Sparse Attention in CPU with LSE exported");
}