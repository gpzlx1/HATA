#include <immintrin.h>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "cpu_attn_ops.h"

#include "cpu_attn.h"

namespace kvlib {

void set_cpu_attn_params(CPUAttnParams &params, torch::Tensor &query,
                         torch::Tensor &key, torch::Tensor &value,
                         torch::Tensor &out, const float scale,
                         const int64_t seqlen_k) {
  params.q_ptr = query.data_ptr();
  params.k_ptr = key.data_ptr();
  params.v_ptr = value.data_ptr();
  params.o_ptr = out.data_ptr();

  params.bsz = query.size(0);
  params.head_dim = query.size(3);
  params.num_kv_heads = key.size(1);
  params.num_heads = query.size(1);
  params.gqa_size = params.num_heads / params.num_kv_heads;

  params.seqlen_q = query.size(2);
  params.seqlen_k = seqlen_k;

  params.q_bsz_stride = query.stride(0);
  params.q_head_stride = query.stride(1);
  params.q_seq_stride = query.stride(2);

  params.k_bsz_stride = key.stride(0);
  params.k_head_stride = key.stride(1);
  params.k_seq_stride = key.stride(2);

  params.v_bsz_stride = value.stride(0);
  params.v_head_stride = value.stride(1);
  params.v_seq_stride = value.stride(2);

  params.o_bsz_stride = out.stride(0);
  params.o_head_stride = out.stride(1);
  params.o_seq_stride = out.stride(2);

  params.scale = scale;
}

void set_lse_params(CPUAttnParams &params, torch::Tensor &lse) {
  params.l_ptr = lse.data_ptr();

  params.return_lse = true;

  params.l_bsz_stride = lse.stride(0);
  params.l_head_stride = lse.stride(1);
  params.l_seq_stride = lse.stride(2);
}

void set_gather_params(CPUSparseAttnParams &params, torch::Tensor &index) {
  params.i_ptr = index.data_ptr();

  params.seqlen_gather = index.size(2);

  params.i_bsz_stride = index.stride(0);
  params.i_head_stride = index.stride(1);
  params.i_seq_stride = index.stride(2);
}

std::vector<torch::Tensor> CPUAttention(torch::Tensor query, torch::Tensor key,
                                        torch::Tensor value, float scale,
                                        int64_t seqlen_k, bool return_lse,
                                        int64_t n_threads) {
  CPUAttnParams params;

  torch::Tensor attn = torch::zeros_like(query);
  set_cpu_attn_params(params, query, key, value, attn, scale, seqlen_k);

  torch::Tensor lse;
  if (return_lse) {
    lse = torch::zeros({query.size(0), query.size(1), query.size(2)},
                       query.options());
    set_lse_params(params, lse);
  }

  int64_t buffer_size = 3 * sizeof(float) * query.size(3) * n_threads +
                        CACHE_LINE_SIZE * n_threads;  // 3x head size/thread
  std::vector<uint8_t> buffer;
  buffer.resize(buffer_size);

#pragma omp parallel num_threads(n_threads)
  {
    ggml_compute_forward_flash_attn_ext_by_thread(params, omp_get_thread_num(),
                                                  n_threads, buffer.data());
  }

  return {attn, lse};
}

std::vector<torch::Tensor> CPUSparseAttention(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor index, float scale, bool return_lse, int64_t n_threads) {
  CPUSparseAttnParams params;

  torch::Tensor attn = torch::zeros_like(query);
  set_cpu_attn_params(params, query, key, value, attn, scale, -1);
  set_gather_params(params, index);

  torch::Tensor lse;
  if (return_lse) {
    lse = torch::zeros({query.size(0), query.size(1), query.size(2)},
                       query.options());
    set_lse_params(params, lse);
  }

  int64_t buffer_size = 3 * sizeof(float) * query.size(3) * n_threads +
                        CACHE_LINE_SIZE * n_threads;  // 3x head size/thread
  std::vector<uint8_t> buffer;
  buffer.resize(buffer_size);

#pragma omp parallel num_threads(n_threads)
  {
    ggml_compute_forward_sparse_flash_attn_ext_by_thread(
        params, omp_get_thread_num(), n_threads, buffer.data());
  }

  return {attn, lse};
}

}  // namespace kvlib