#pragma once

#include <torch/script.h>
#include <cstdint>
#include <vector>

namespace kvlib {

struct CPUAttnParams {
  using index_t = int32_t;

  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  void *o_ptr;
  void *l_ptr = nullptr;

  index_t bsz;
  index_t head_dim;
  index_t num_heads;
  index_t num_kv_heads;
  index_t seqlen_q;
  index_t seqlen_k;

  index_t q_bsz_stride;
  index_t q_head_stride;
  index_t q_seq_stride;

  index_t k_bsz_stride;
  index_t k_head_stride;
  index_t k_seq_stride;

  index_t v_bsz_stride;
  index_t v_head_stride;
  index_t v_seq_stride;

  index_t o_bsz_stride;
  index_t o_head_stride;
  index_t o_seq_stride;

  index_t l_bsz_stride;
  index_t l_head_stride;
  index_t l_seq_stride;

  bool return_lse = false;
  float scale;
  int32_t gqa_size;
};

struct CPUSparseAttnParams : public CPUAttnParams {
  void *i_ptr;

  index_t seqlen_gather;

  index_t i_bsz_stride;
  index_t i_head_stride;
  index_t i_seq_stride;
};


#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

typedef uint8_t half;

void ggml_compute_forward_flash_attn_ext_by_thread(CPUAttnParams &params,
                                                   const int ith, const int nth,
                                                   uint8_t *wdata);
void ggml_compute_forward_sparse_flash_attn_ext_by_thread(
    CPUSparseAttnParams &params, const int ith, const int nth, uint8_t *wdata);

}  // namespace kvlib