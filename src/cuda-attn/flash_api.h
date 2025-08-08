/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <cutlass/numeric_types.h>

#include "include/flash.h"
#include "include/static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

namespace kvlib {

//////////////////////////////////////////////////////////////////////////////////

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const at::Tensor q, const at::Tensor k,
                      const at::Tensor v, at::Tensor out, void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d, void *seqused_k, void *p_d,
                      void *softmax_lse_d, float p_dropout, float softmax_scale,
                      int window_size_left, int window_size_right,
                      const float softcap, bool seqlenq_ngroups_swapped = false,
                      const bool unpadded_lse = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int *>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

// Set the different scale values.
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(softcap <= 0.0,
              "This flash attention build does not support softcap.");
#endif
  if (softcap > 0.0) {
    params.softcap = softmax_scale / softcap;
    params.scale_softmax = softcap;
    params.scale_softmax_log2 = softcap * M_LOG2E;
  } else {
    // Remove potential NaN
    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  }

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of < params.p_dropout_in_uint =
  // uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout *
  // 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f,
              "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      params.is_causal || (window_size_left < 0 && window_size_right < 0),
      "This flash attention build does not support local attention.");
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded,
              "This flash attention build does not support headdim not being a "
              "multiple of 32.");
#endif

  params.unpadded_lse = unpadded_lse;
  params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) !=
                                  ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      return num_splits;
    }
  }
  return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(
    Flash_fwd_params &params, const int batch_size, const int num_heads,
    const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout, const int num_splits,
    cudaDeviceProp *dprops, struct c10::TensorOptions opts) {
  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard
  // kernel. In any case we don't expect seqlen_q to be larger than 64 for
  // inference.
  const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
  params.num_splits = num_splits;
  at::Tensor softmax_lse_accum;
  at::Tensor out_accum;

  if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
    if (num_splits < 1) {
      // We multiply number of SMs by 2 to hard-code the fact that we're using
      // 128 threads per block.
      params.num_splits = num_splits_heuristic(
          batch_size * num_heads * num_m_blocks,
          dprops->multiProcessorCount * 2, num_n_blocks, 128);
    }
    if (params.num_splits > 1) {
      softmax_lse_accum =
          torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q},
                       opts.dtype(at::kFloat));
      out_accum = torch::empty({params.num_splits, batch_size, num_heads,
                                max_seqlen_q, head_size_rounded},
                               opts.dtype(at::kFloat));
      params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
      params.oaccum_ptr = out_accum.data_ptr();
    }
    TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
  }

  return std::make_tuple(softmax_lse_accum, out_accum);
}

void set_params_gather(Flash_fwd_params &params, const at::Tensor idx,
                       const int num_heads_gather,
                       const int seqlen_gather_rounded) {
  params.gather_idx_ptr = idx.data_ptr();
  params.gather_idx_batch_stride = idx.stride(0);
  params.gather_idx_head_stride = idx.stride(1);
  params.num_heads_gather = num_heads_gather;
  params.seqlen_gather = idx.size(2);
  params.seqlen_gather_rounded = seqlen_gather_rounded;
}

void set_params_mixed(Flash_fwd_params &params, const at::Tensor idx,
                      const at::Tensor buffer_keys,
                      const at::Tensor buffer_values,
                      const at::Tensor k_head_mask,
                      const at::Tensor k_head_index) {
  params.gather_idx_ptr = idx.data_ptr();
  params.gather_idx_batch_stride = idx.stride(0);
  params.gather_idx_head_stride = idx.stride(1);

  params.k_head_mask = k_head_mask.data_ptr();
  params.k_head_index = k_head_index.data_ptr();
  params.buffer_k = buffer_keys.data_ptr();
  params.buffer_v = buffer_values.data_ptr();

  params.buffer_k_batch_stride = buffer_keys.stride(0);
  params.buffer_v_batch_stride = buffer_values.stride(0);
  params.buffer_k_row_stride = buffer_keys.stride(1);
  params.buffer_v_row_stride = buffer_values.stride(1);
  params.buffer_k_head_stride = buffer_keys.stride(2);
  params.buffer_v_head_stride = buffer_values.stride(2);
}

//////////////////////////////////////////////////////////////////////////////////

void run_mha_gather_fwd(Flash_fwd_params &params, cudaStream_t stream,
                        bool force_split_kernel = false) {
  if (params.num_splits <= 1 &&
      !force_split_kernel) {  // If we don't set it num_splits == 0
    run_mha_gather_fwd_<cutlass::half_t, 128, false>(params, stream);
  } else {
    run_mha_gather_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(params,
                                                                     stream);
  }
}

std::vector<at::Tensor> mha_index_decode_fwd(
    at::Tensor
        &q,  // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,    // batch_size x seqlen_k x num_heads_k x
                            // round_multiple(head_size, 8)
    const at::Tensor &v,    // batch_size x seqlen_k x num_heads_k x
                            // round_multiple(head_size, 8)
    const at::Tensor &idx,  // batch_size x num_heads[_k] x seqlen_gather
    const float softmax_scale) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x,
              "FlashAttention only supports Ampere GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_sm90 || is_sm8x,
                "bfloat16 is only supported on Ampere GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  const int num_heads_gather = idx.size(1);
  const int seqlen_gather = idx.size(2);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  const int seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0 &&
      num_heads_gather == num_heads_k;
  const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
            .transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out = torch::empty_like(q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
  const int seqlen_gather_rounded = round_multiple(seqlen_gather, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  Flash_fwd_params params;
  set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded, q, k, v, out, nullptr, nullptr, nullptr,
                   nullptr, softmax_lse.data_ptr(), 0.0f, softmax_scale, -1, -1,
                   0.0f);

  set_params_gather(params, idx, num_heads_gather, seqlen_gather_rounded);

  // Keep references to these tensors to extend their lifetime
  at::Tensor softmax_lse_accum, out_accum;
  std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
      params, batch_size, num_heads, head_size, seqlen_gather, seqlen_q,
      head_size_rounded, 0.0f, /*num_splits=*/0, dprops, opts);

  if (seqlen_gather > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_gather_fwd(params, stream);
  } else {
    out.zero_();
  }

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }
  return {out, softmax_lse};
}

//////////////////////////////////////////////////////////////////////////////////

void run_mha_mixed_fwd(Flash_fwd_params &params, cudaStream_t stream,
                       bool force_split_kernel = false) {
  if (params.num_splits <= 1 &&
      !force_split_kernel) {  // If we don't set it num_splits == 0
    run_mha_mixed_fwd_<cutlass::half_t, 128, false>(params, stream);
  } else {
    run_mha_mixed_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(params,
                                                                    stream);
  }
}

std::vector<at::Tensor> mha_mixed_decode_fwd(
    at::Tensor
        &q,  // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &cached_k,  // batch_size x (>real_k_seq) x (<num_heads_k>)
                                 // x round_multiple(head_size, 8)
    const at::Tensor &cached_v,  // batch_size x (>real_k_seq) x (<num_heads_k>)
                                 // x round_multiple(head_size, 8)
    const at::Tensor &gather_idx,  // batch_size x num_heads_k x seqlen_gather
    const at::Tensor
        &buffer_k,  // batch_size x (>real_k_seq) x (<num_heads_k>) x head_size
    const at::Tensor
        &buffer_v,  // batch_size x (>real_k_seq) x (<num_heads_k>) x head_size
    const at::Tensor &k_head_mask,   // num_heads_k
    const at::Tensor &k_head_index,  // num_heads_K
    const int real_k_seq, const float softmax_scale) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x,
              "FlashAttention only supports Ampere GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_sm90 || is_sm8x,
                "bfloat16 is only supported on Ampere GPUs or newer");
  }
  TORCH_CHECK(cached_k.dtype() == q_dtype,
              "query and key must have the same dtype");
  TORCH_CHECK(cached_v.dtype() == q_dtype,
              "query and value must have the same dtype");
  TORCH_CHECK(buffer_k.dtype() == q_dtype,
              "query and key must have the same dtype");
  TORCH_CHECK(buffer_v.dtype() == q_dtype,
              "query and value must have the same dtype");

  // int check for k_head_index and gather_idx
  TORCH_CHECK(gather_idx.dtype() == torch::kInt32, "gather_idx must be int32");
  TORCH_CHECK(k_head_index.dtype() == torch::kInt32,
              "k_head_index must be int32");
  TORCH_CHECK(k_head_mask.dtype() == torch::kBool, "k_head_mask must be bool");

  CHECK_DEVICE(q);
  CHECK_DEVICE(cached_k);
  CHECK_DEVICE(cached_k);
  CHECK_DEVICE(buffer_k);
  CHECK_DEVICE(buffer_v);
  CHECK_DEVICE(k_head_mask);
  CHECK_DEVICE(k_head_index);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(cached_k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(cached_k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(buffer_k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(buffer_v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = real_k_seq;
  // const int num_heads_k = cached_k.size(2);
  const int num_heads_k = k_head_mask.size(0);

  // const int num_heads_gather = gather_idx.size(1);
  // const int seqlen_gather = gather_idx.size(2);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  TORCH_CHECK(num_heads_k == gather_idx.size(1));
  TORCH_CHECK(num_heads_k == k_head_index.size(0));

  TORCH_CHECK(seqlen_k == gather_idx.size(2));
  TORCH_CHECK(seqlen_k <= buffer_k.size(1));

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  const int seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0; // &&
      // num_heads_gather == num_heads_k;
      const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
            .transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  // CHECK_SHAPE(cached_k, batch_size, seqlen_k, num_heads_k, head_size);
  // CHECK_SHAPE(cached_v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out = torch::empty_like(q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
  // const int seqlen_gather_rounded = round_multiple(seqlen_gather, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  Flash_fwd_params params;
  set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded, q, cached_k, cached_v, out, nullptr,
                   nullptr, nullptr, nullptr, softmax_lse.data_ptr(), 0.0f,
                   softmax_scale, -1, -1, 0.0f);

  // set_params_gather(params, gather_idx, num_heads_gather,
  // seqlen_gather_rounded);
  set_params_mixed(params, gather_idx, buffer_k, buffer_v, k_head_mask,
                   k_head_index);

  // Keep references to these tensors to extend their lifetime
  at::Tensor softmax_lse_accum, out_accum;
  std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
      params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
      head_size_rounded, 0.0f, /*num_splits=*/0, dprops, opts);

  if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_mixed_fwd(params, stream);
  } else {
    out.zero_();
  }

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }
  return {out, softmax_lse};
}

//////////////////////////////////////////////////////////////////////////////////

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream,
                 bool force_split_kernel = false) {
  if (params.num_splits <= 1 &&
      !force_split_kernel) {  // If we don't set it num_splits == 0
    run_mha_fwd_<cutlass::half_t, 128, false>(params, stream);
  } else {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(params, stream);
  }
}

std::vector<at::Tensor> mha_decode_fwd(
    at::Tensor
        &q,  // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,  // batch_size x ? x num_heads_k x
                          // round_multiple(head_size, 8)
    const at::Tensor &v,  // batch_size x ? x num_heads_k x
                          // round_multiple(head_size, 8)
    const float softmax_scale, const int32_t seqlen_k) {
  auto dprops = at::cuda::getCurrentDeviceProperties();

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int num_heads_k = k.size(2);

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  const int seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0;
  const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
            .transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  at::Tensor out = torch::empty_like(q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  Flash_fwd_params params;
  set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded, q, k, v, out, nullptr, nullptr, nullptr,
                   nullptr, softmax_lse.data_ptr(), 0.0f, softmax_scale, -1, -1,
                   0.0f);

  // Keep references to these tensors to extend their lifetime
  at::Tensor softmax_lse_accum, out_accum;
  std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
      params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
      head_size_rounded, 0.0f, /*num_splits=*/0, dprops, opts);

  if (seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
  } else {
    out.zero_();
  }

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }
  return {out, softmax_lse};
}

//////////////////////////////////////////////////////////////////////////////////

}  // namespace kvlib