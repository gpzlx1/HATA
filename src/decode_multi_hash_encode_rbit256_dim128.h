#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload__decode_multi_hash_encode_rbit256_dim128_kernel_17940b89_0123456789101112131415(void);
void load__decode_multi_hash_encode_rbit256_dim128_kernel_17940b89_0123456789101112131415(void);
// tt-linker: _decode_multi_hash_encode_rbit256_dim128_kernel_17940b89_0123456789101112131415:CUdeviceptr key_data_ptr, int64_t key_data_stride0, CUdeviceptr query_data_ptr, int64_t query_data_stride0, CUdeviceptr hash_weight_ptr, CUdeviceptr packbit_tensor_ptr, CUdeviceptr key_code_output_ptr, int64_t key_code_output_stride0, CUdeviceptr key_norm_output_ptr, int64_t key_norm_output_stride0, CUdeviceptr query_code_output_ptr, int64_t query_code_output_stride0, int32_t CUR_SEQ, int32_t BSZ, int32_t KV_HEAD, int32_t Q_HEAD:256x128x16_warps4xstages1
CUresult _decode_multi_hash_encode_rbit256_dim128_kernel_17940b89_0123456789101112131415(CUstream stream, CUdeviceptr key_data_ptr, int64_t key_data_stride0, CUdeviceptr query_data_ptr, int64_t query_data_stride0, CUdeviceptr hash_weight_ptr, CUdeviceptr packbit_tensor_ptr, CUdeviceptr key_code_output_ptr, int64_t key_code_output_stride0, CUdeviceptr key_norm_output_ptr, int64_t key_norm_output_stride0, CUdeviceptr query_code_output_ptr, int64_t query_code_output_stride0, int32_t CUR_SEQ, int32_t BSZ, int32_t KV_HEAD, int32_t Q_HEAD);
