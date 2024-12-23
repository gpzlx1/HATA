#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload__combine_attention_dim128_kernel_49e4dd2f_012345(void);
void load__combine_attention_dim128_kernel_49e4dd2f_012345(void);
// tt-linker: _combine_attention_dim128_kernel_49e4dd2f_012345:CUdeviceptr
// attn_a_ptr, CUdeviceptr attn_b_ptr, CUdeviceptr lse_a_ptr, CUdeviceptr
// lse_b_ptr, CUdeviceptr output_ptr, int32_t TOTAL_HEAD:128x8_warps4xstages1
CUresult _combine_attention_dim128_kernel_49e4dd2f_012345(
    CUstream stream, CUdeviceptr attn_a_ptr, CUdeviceptr attn_b_ptr,
    CUdeviceptr lse_a_ptr, CUdeviceptr lse_b_ptr, CUdeviceptr output_ptr,
    int32_t TOTAL_HEAD);
