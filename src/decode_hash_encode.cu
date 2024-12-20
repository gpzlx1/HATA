#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

extern "C" {
#include "decode_hash_encode_rbit128_dim128.h"
#include "decode_hash_encode_rbit256_dim128.h"
}

#include "operator.h"

namespace kvlib {

void decode_hash_encode(torch::Tensor key_data, torch::Tensor hash_weights,
                        torch::Tensor key_code_output,
                        torch::Tensor key_norm_output, torch::Tensor query_data,
                        torch::Tensor query_code_output,
                        torch::Tensor packbit_aux_tensor, int32_t cur_seq) {
  const int32_t rbit = hash_weights.size(1);
  CHECK(rbit % 32 == 0);

  const int32_t bsz = key_data.size(0);
  const int32_t seq = key_data.size(1);
  const int32_t num_kv_head = key_data.size(2);
  const int32_t head_dim = key_data.size(3);

  const int32_t num_head = query_data.size(2);

  CHECK(seq == 1);
  CHECK(head_dim == 128);

  auto device = key_data.device();
  int32_t device_id = device.index();

  CUstream stream = (CUstream)c10::cuda::getCurrentCUDAStream(device_id);
  CUdeviceptr key_data_ptr = (CUdeviceptr)key_data.data_ptr<at::Half>();
  int64_t key_data_stride0 = key_data.stride(0);
  CUdeviceptr query_data_ptr = (CUdeviceptr)query_data.data_ptr<at::Half>();
  int64_t query_data_stride0 = query_data.stride(0);
  CUdeviceptr hash_weight_ptr = (CUdeviceptr)hash_weights.data_ptr<at::Half>();
  CUdeviceptr packbit_tensor_ptr =
      (CUdeviceptr)packbit_aux_tensor.data_ptr<int32_t>();
  CUdeviceptr key_code_output_ptr =
      (CUdeviceptr)key_code_output.data_ptr<int32_t>();
  int64_t key_code_output_stride0 = key_code_output.stride(0);
  CUdeviceptr key_norm_output_ptr =
      (CUdeviceptr)key_norm_output.data_ptr<at::Half>();
  int64_t key_norm_output_stride0 = key_norm_output.stride(0);
  CUdeviceptr query_code_output_ptr =
      (CUdeviceptr)query_code_output.data_ptr<int32_t>();
  int64_t query_code_output_stride0 = query_code_output.stride(0);
  int32_t CUR_SEQ = cur_seq;
  int32_t BSZ = bsz;
  int32_t MIN_TOTAL_HAED = num_kv_head;
  int32_t MAX_TOTAL_HEAD = num_head;

  if (rbit == 128) {
    _decode_hash_encode_rbit128_dim128_kernel_a44367f2_0123456789101112131415(
        stream, key_data_ptr, key_data_stride0, query_data_ptr,
        query_data_stride0, hash_weight_ptr, packbit_tensor_ptr,
        key_code_output_ptr, key_code_output_stride0, key_norm_output_ptr,
        key_norm_output_stride0, query_code_output_ptr,
        query_code_output_stride0, CUR_SEQ, BSZ, MIN_TOTAL_HAED,
        MAX_TOTAL_HEAD);
  } else if (rbit == 256) {
    _decode_hash_encode_rbit256_dim128_kernel_17940b89_0123456789101112131415(
        stream, key_data_ptr, key_data_stride0, query_data_ptr,
        query_data_stride0, hash_weight_ptr, packbit_tensor_ptr,
        key_code_output_ptr, key_code_output_stride0, key_norm_output_ptr,
        key_norm_output_stride0, query_code_output_ptr,
        query_code_output_stride0, CUR_SEQ, BSZ, MIN_TOTAL_HAED,
        MAX_TOTAL_HEAD);
  } else {
    CHECK(false);
  }
}

}  // namespace kvlib