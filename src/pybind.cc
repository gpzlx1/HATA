#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "attn/flash_api.h"
#include "operator.h"

namespace py = pybind11;

PYBIND11_MODULE(KVLib, m) {
  m.def("hamming_score", &kvlib::HammingScoreCUDA)
      .def("batch_topk", &kvlib::TopkCUDA)
      .def("encode", &kvlib::EncodeCUDA)
      .def("flash_index_decode", &mha_index_decode_fwd);
}