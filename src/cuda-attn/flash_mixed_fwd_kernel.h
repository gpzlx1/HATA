/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "include/kernel_traits.h"

#include "include/block_info.h"
#include "include/mask.h"
#include "include/softmax.h"
#include "include/utils.h"

#include "flash_common_kernel.h"

namespace kvlib {
namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_dropout, bool Is_causal,
          bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_mixed_attn_1rowblock_cached_kv_with_gather(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int real_kv_bidh) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_block_min =
      !Is_local
          ? 0
          : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k -
                         binfo.actual_seqlen_q - params.window_size_left) /
                            kBlockN);
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                           binfo.actual_seqlen_q + params.window_size_right,
                       kBlockN));
  }
  // We exit early and write 0 to gO and gLSE. This also covers the case where
  // actual_seqlen_k == 0. Otherwise we might read OOB elements from gK and gV.
  if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<Element *>(params.o_ptr) +
            binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
        params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
                /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgO); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM &&
          get<1>(tOcO(0, m, 0)) == 0) {
        gLSE(row) = INFINITY;
      }
    }
    return;
  }

  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                                binfo.q_offset(params.q_batch_stride,
                                               params.q_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)

  // k_buff and v_buff is for cached_kv with gather
  int *gather_idx_buff = reinterpret_cast<int *>(params.gather_idx_ptr) +
                         binfo.idx_offset(params.gather_idx_batch_stride,
                                          params.gather_idx_head_stride, bidb,
                                          bidh / params.h_h_k_ratio);
  Element *k_buff = nullptr;
  Element *v_buff = nullptr;

  k_buff = reinterpret_cast<Element *>(params.k_ptr) +
           binfo.k_offset(params.k_batch_stride, params.k_row_stride,
                          params.k_head_stride, bidb, real_kv_bidh);
  v_buff = reinterpret_cast<Element *>(params.v_ptr) +
           binfo.k_offset(params.v_batch_stride, params.v_row_stride,
                          params.v_head_stride, bidb, real_kv_bidh);

  // others
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data().get(),
                  typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  //
  // PREDICATES
  //

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
  }

  //
  // Prologue
  //

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                                     tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).
  int n_block = n_block_max - 1;

  flash::gather_g2s_copy<Kernel_traits>(
      gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
      params.k_row_stride, n_block, tidx,
      binfo.actual_seqlen_k - n_block * kBlockN);

  cute::cp_async_fence();

  clear(acc_o);

  flash::Softmax<2 * size<1>(acc_o)> softmax;

  const float alibi_slope =
      !Has_alibi || params.alibi_slopes_ptr == nullptr
          ? 0.0f
          : reinterpret_cast<float *>(params.alibi_slopes_ptr)
                    [bidb * params.alibi_slopes_batch_stride + bidh] /
                params.scale_softmax;
  flash::Mask<Is_causal, Is_local, Has_alibi> mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left,
      params.window_size_right, 0.0f);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that
  // case we need to mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps =
      (!Is_causal && !Is_local)
          ? 1
          : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                       : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      flash::gather_g2s_copy<Kernel_traits>(gmem_tiled_copy_QKV, params, tVsV,
                                            v_buff, gather_idx_buff,
                                            params.v_row_stride, n_block, tidx);

    } else {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tVsV, v_buff, gather_idx_buff,
          params.v_row_stride, n_block, tidx,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    flash::cp_async_wait<0>();
    __syncthreads();

    if (n_block > n_block_min) {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
          params.k_row_stride, n_block - 1, tidx);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    masking_step == 0
        ? softmax.template softmax_rescale_o<
              /*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(
              acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o<
              /*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
              acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    flash::gather_g2s_copy<Kernel_traits>(gmem_tiled_copy_QKV, params, tVsV,
                                          v_buff, gather_idx_buff,
                                          params.v_row_stride, n_block, tidx);

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
          params.k_row_stride, n_block - 1, tidx);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    mask.template apply_mask</*Causal_mask=*/false>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    softmax
        .template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(
            acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);
  }

  // Epilogue

  Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
      acc_o, params.scale_softmax);

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) +
                                binfo.q_offset(params.o_batch_stride,
                                               params.o_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)
  Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
      params, bidb, bidh, m_block, binfo);

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split,
          bool Append_KV, typename Params>
inline __device__ void
compute_mixed_attn_1rowblock_splitkv_cached_kv_with_gather(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int n_split_idx, const int num_n_splits, const int real_kv_bidh) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  using GmemTiledCopyO =
      std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO,
                         typename Kernel_traits::GmemTiledCopyOaccum>;
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_blocks_per_split =
      ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) /
      num_n_splits;
  const int n_block_min =
      !Is_local ? n_split_idx * n_blocks_per_split
                : std::max(n_split_idx * n_blocks_per_split,
                           (m_block * kBlockM + binfo.actual_seqlen_k -
                            binfo.actual_seqlen_q - params.window_size_left) /
                               kBlockN);
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN),
                             (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                           binfo.actual_seqlen_q + params.window_size_right,
                       kBlockN));
  }

  if (n_block_min >=
      n_block_max) {  // This also covers the case where n_block_max <= 0
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
    // Otherwise we might read OOB elements from gK and gV,
    // or get wrong results when we combine gOaccum from different blocks.
    const index_t row_offset_o =
        binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
        m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum =
        (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
         m_block * kBlockM) *
        params.d_rounded;
    const index_t row_offset_lseaccum =
        ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
        m_block * kBlockM;
    Tensor gOaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr
                                                         : params.o_ptr) +
                      (Split ? row_offset_oaccum : row_offset_o)),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr
                                                   : params.softmax_lse_ptr) +
            row_offset_lseaccum),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    clear(tOrOaccum);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gOaccum), size<1>(gOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
                /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgOaccum); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM &&
          get<1>(tOcO(0, m, 0)) == 0) {
        gLSEaccum(row) = Split ? -INFINITY : INFINITY;
      }
    }
    return;
  }

  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                                binfo.q_offset(params.q_batch_stride,
                                               params.q_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)

  // k_buff, v_buff and gather_idx_buff is for cached_kv with gather
  int *gather_idx_buff = reinterpret_cast<int *>(params.gather_idx_ptr) +
                         binfo.idx_offset(params.gather_idx_batch_stride,
                                          params.gather_idx_head_stride, bidb,
                                          bidh / params.h_h_k_ratio);
  Element *k_buff = nullptr;
  Element *v_buff = nullptr;

  k_buff = reinterpret_cast<Element *>(params.k_ptr) +
           binfo.k_offset(params.k_batch_stride, params.k_row_stride,
                          params.k_head_stride, bidb, real_kv_bidh);
  v_buff = reinterpret_cast<Element *>(params.v_ptr) +
           binfo.k_offset(params.v_batch_stride, params.v_row_stride,
                          params.v_head_stride, bidb, real_kv_bidh);

  // others
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data().get(),
                  typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // PREDICATES

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
  }
  // Prologue

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                                     tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);

  // We iterate over the blocks in reverse order. This is because the last
  // block is the only one that needs masking when we read K and V from global
  // memory. Moreover, iterating in reverse might save us 1 register (we just
  // need n_block instead of both n_block and n_block_max).
  int n_block = n_block_max - 1;

  flash::gather_g2s_copy<Kernel_traits>(
      gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
      params.k_row_stride, n_block, tidx,
      binfo.actual_seqlen_k - n_block * kBlockN);

  cute::cp_async_fence();

  clear(acc_o);

  flash::Softmax<2 * size<1>(acc_o)> softmax;

  const float alibi_slope =
      !Has_alibi ? 0.0f
                 : reinterpret_cast<float *>(params.alibi_slopes_ptr)
                           [bidb * params.alibi_slopes_batch_stride + bidh] /
                       params.scale_softmax;
  flash::Mask<Is_causal, Is_local, Has_alibi> mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left,
      params.window_size_right, alibi_slope);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the
  // last ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that
  // case we need to mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps =
      (!Is_causal && !Is_local)
          ? 1
          : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                       : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      flash::gather_g2s_copy<Kernel_traits>(gmem_tiled_copy_QKV, params, tVsV,
                                            v_buff, gather_idx_buff,
                                            params.v_row_stride, n_block, tidx);

    } else {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tVsV, v_buff, gather_idx_buff,
          params.v_row_stride, n_block, tidx,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    flash::cp_async_wait<0>();
    __syncthreads();

    if (n_block > n_block_min) {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
          params.k_row_stride, n_block - 1, tidx);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // We have key_padding_mask so we'll need to Check_inf
    masking_step == 0
        ? softmax.template softmax_rescale_o</*Is_first=*/true,
                                             /*Check_inf=*/Is_causal ||
                                                 Is_local || !Is_even_MN>(
              acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o</*Is_first=*/false,
                                             /*Check_inf=*/Is_causal ||
                                                 Is_local || !Is_even_MN>(
              acc_s, acc_o, params.scale_softmax_log2);
    // if (cute::thread0()) { print(scores_max); print(scores_sum);
    // print(scores); }

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    flash::gather_g2s_copy<Kernel_traits>(gmem_tiled_copy_QKV, params, tVsV,
                                          v_buff, gather_idx_buff,
                                          params.v_row_stride, n_block, tidx);

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      flash::gather_g2s_copy<Kernel_traits>(
          gmem_tiled_copy_QKV, params, tKsK, k_buff, gather_idx_buff,
          params.k_row_stride, n_block - 1, tidx);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    mask.template apply_mask</*Causal_mask=*/false>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
    softmax.template softmax_rescale_o</*Is_first=*/false,
                                       /*Check_inf=*/Is_local>(
        acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);
  }

  // Epilogue

  Tensor lse =
      softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
          acc_o, params.scale_softmax);
  // if (cute::thread0()) { print(lse); }

  Tensor sOaccum =
      make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)),
                  typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  using SmemTiledCopyO =
      std::conditional_t<!Split, typename Kernel_traits::SmemCopyAtomO,
                         typename Kernel_traits::SmemCopyAtomOaccum>;
  auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
  auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor rO = flash::convert_type<ElementO>(acc_o);
  Tensor taccOrOaccum =
      smem_thr_copy_Oaccum.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(
      sOaccum);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sOaccum is larger than sQ, so we need to syncthreads here
  // TODO: allocate enough smem for sOaccum
  if constexpr (Split) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_oaccum =
      (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
       m_block * kBlockM) *
      params.d_rounded;
  const index_t row_offset_lseaccum =
      (Split || !params.unpadded_lse
           ? ((n_split_idx * params.b + bidb) * params.h + bidh) *
                 params.seqlen_q
           : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)) +
      m_block * kBlockM;

  Tensor gOaccum =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(
                                    Split ? params.oaccum_ptr : params.o_ptr) +
                                (Split ? row_offset_oaccum : row_offset_o)),
                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                  make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
  Tensor gLSEaccum = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr
                                                 : params.softmax_lse_ptr) +
          row_offset_lseaccum),
      Shape<Int<kBlockM>>{}, Stride<_1>{});
  // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n",
  // row_offset_o, bidh, gOaccum.data()); }

  GmemTiledCopyO gmem_tiled_copy_Oaccum;
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(
      sOaccum);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

  __syncthreads();

  Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
  cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(
      size<0>(sOaccum), size<1>(sOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(
      cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_Oaccum, tOrOaccum,
                                     tOgOaccum, tOcO, tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_dropout, bool Is_causal,
          bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_mixed_attn_1rowblock_buffer_kv(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int real_kv_bidh) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_block_min =
      !Is_local
          ? 0
          : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k -
                         binfo.actual_seqlen_q - params.window_size_left) /
                            kBlockN);
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                           binfo.actual_seqlen_q + params.window_size_right,
                       kBlockN));
  }
  // We exit early and write 0 to gO and gLSE. This also covers the case where
  // actual_seqlen_k == 0. Otherwise we might read OOB elements from gK and gV.
  if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<Element *>(params.o_ptr) +
            binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
        params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
                /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgO); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM &&
          get<1>(tOcO(0, m, 0)) == 0) {
        gLSE(row) = INFINITY;
      }
    }
    return;
  }

  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                                binfo.q_offset(params.q_batch_stride,
                                               params.q_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)

  // mK, gK, mV, gV is for buffer_kv
  Tensor mK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.buffer_k) +
                    binfo.k_offset(params.buffer_k_batch_stride,
                                   params.buffer_k_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k,
                 params.d),  // todo use real buffer k head in the future
      make_stride(params.buffer_k_row_stride, params.buffer_k_head_stride,
                  _1{}));
  Tensor gK =
      local_tile(mK(_, real_kv_bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                 make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
  Tensor mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.buffer_v) +
                    binfo.k_offset(params.buffer_v_batch_stride,
                                   params.buffer_v_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k,
                 params.d),  // todo use real buffer k head in the future
      make_stride(params.buffer_v_row_stride, params.buffer_v_head_stride,
                  _1{}));
  Tensor gV =
      local_tile(mV(_, real_kv_bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                 make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

  // others
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data().get(),
                  typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // for buffer_kv
  Tensor tKgK =
      gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
  Tensor tVgV =
      gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  //
  // PREDICATES
  //

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  //
  // Prologue
  //

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                                     tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).
  int n_block = n_block_max - 1;

  flash::copy<Is_even_MN, Is_even_K>(
      gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
      binfo.actual_seqlen_k - n_block * kBlockN);

  cute::cp_async_fence();

  clear(acc_o);

  flash::Softmax<2 * size<1>(acc_o)> softmax;

  const float alibi_slope =
      !Has_alibi || params.alibi_slopes_ptr == nullptr
          ? 0.0f
          : reinterpret_cast<float *>(params.alibi_slopes_ptr)
                    [bidb * params.alibi_slopes_batch_stride + bidh] /
                params.scale_softmax;
  flash::Mask<Is_causal, Is_local, Has_alibi> mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left,
      params.window_size_right, 0.0f);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that
  // case we need to mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps =
      (!Is_causal && !Is_local)
          ? 1
          : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                       : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

    } else {
      flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    flash::cp_async_wait<0>();
    __syncthreads();

    if (n_block > n_block_min) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV,
                                                  tKgK(_, _, _, n_block - 1),
                                                  tKsK, tKVcKV, tKVpKV);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    masking_step == 0
        ? softmax.template softmax_rescale_o<
              /*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(
              acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o<
              /*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
              acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    flash::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV,
                                                  tKgK(_, _, _, n_block - 1),
                                                  tKsK, tKVcKV, tKVpKV);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    mask.template apply_mask</*Causal_mask=*/false>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    softmax
        .template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(
            acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);
  }

  // Epilogue

  Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
      acc_o, params.scale_softmax);

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) +
                                binfo.q_offset(params.o_batch_stride,
                                               params.o_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)
  Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
      params, bidb, bidh, m_block, binfo);

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split,
          bool Append_KV, typename Params>
inline __device__ void compute_mixed_attn_1rowblock_splitkv_buffer_kv(
    const Params &params, const int bidb, const int bidh, const int m_block,
    const int n_split_idx, const int num_n_splits, const int real_kv_bidh) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  using GmemTiledCopyO =
      std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO,
                         typename Kernel_traits::GmemTiledCopyOaccum>;
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_blocks_per_split =
      ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) /
      num_n_splits;
  const int n_block_min =
      !Is_local ? n_split_idx * n_blocks_per_split
                : std::max(n_split_idx * n_blocks_per_split,
                           (m_block * kBlockM + binfo.actual_seqlen_k -
                            binfo.actual_seqlen_q - params.window_size_left) /
                               kBlockN);
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN),
                             (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                           binfo.actual_seqlen_q + params.window_size_right,
                       kBlockN));
  }

  if (n_block_min >=
      n_block_max) {  // This also covers the case where n_block_max <= 0
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
    // Otherwise we might read OOB elements from gK and gV,
    // or get wrong results when we combine gOaccum from different blocks.
    const index_t row_offset_o =
        binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
        m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum =
        (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
         m_block * kBlockM) *
        params.d_rounded;
    const index_t row_offset_lseaccum =
        ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
        m_block * kBlockM;
    Tensor gOaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr
                                                         : params.o_ptr) +
                      (Split ? row_offset_oaccum : row_offset_o)),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr
                                                   : params.softmax_lse_ptr) +
            row_offset_lseaccum),
        Shape<Int<kBlockM>>{}, Stride<_1>{});

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    clear(tOrOaccum);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gOaccum), size<1>(gOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
                /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgOaccum); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM &&
          get<1>(tOcO(0, m, 0)) == 0) {
        gLSEaccum(row) = Split ? -INFINITY : INFINITY;
      }
    }
    return;
  }

  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                                binfo.q_offset(params.q_batch_stride,
                                               params.q_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)

  // mK, gK, mV, gV is for buffer_kv
  Tensor mK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.buffer_k) +
                    binfo.k_offset(params.buffer_k_batch_stride,
                                   params.buffer_k_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k,
                 params.d),  // todo use real buffer k head in the future
      make_stride(params.buffer_k_row_stride, params.buffer_k_head_stride,
                  _1{}));
  Tensor gK =
      local_tile(mK(_, real_kv_bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                 make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
  Tensor mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.buffer_v) +
                    binfo.k_offset(params.buffer_v_batch_stride,
                                   params.buffer_v_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k,
                 params.d),  // todo use real buffer k head in the future
      make_stride(params.buffer_v_row_stride, params.buffer_v_head_stride,
                  _1{}));
  Tensor gV =
      local_tile(mV(_, real_kv_bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                 make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

  // others
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data().get(),
                  typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // for buffer_kv
  Tensor tKgK =
      gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
  Tensor tVgV =
      gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // PREDICATES

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }
  // Prologue

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                                     tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);

  // We iterate over the blocks in reverse order. This is because the last
  // block is the only one that needs masking when we read K and V from global
  // memory. Moreover, iterating in reverse might save us 1 register (we just
  // need n_block instead of both n_block and n_block_max).
  int n_block = n_block_max - 1;

  flash::copy<Is_even_MN, Is_even_K>(
      gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
      binfo.actual_seqlen_k - n_block * kBlockN);

  cute::cp_async_fence();

  clear(acc_o);

  flash::Softmax<2 * size<1>(acc_o)> softmax;

  const float alibi_slope =
      !Has_alibi ? 0.0f
                 : reinterpret_cast<float *>(params.alibi_slopes_ptr)
                           [bidb * params.alibi_slopes_batch_stride + bidh] /
                       params.scale_softmax;
  flash::Mask<Is_causal, Is_local, Has_alibi> mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left,
      params.window_size_right, alibi_slope);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the
  // last ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that
  // case we need to mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps =
      (!Is_causal && !Is_local)
          ? 1
          : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                       : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

    } else {
      flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    flash::cp_async_wait<0>();
    __syncthreads();

    if (n_block > n_block_min) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV,
                                                  tKgK(_, _, _, n_block - 1),
                                                  tKsK, tKVcKV, tKVpKV);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // We have key_padding_mask so we'll need to Check_inf
    masking_step == 0
        ? softmax.template softmax_rescale_o</*Is_first=*/true,
                                             /*Check_inf=*/Is_causal ||
                                                 Is_local || !Is_even_MN>(
              acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o</*Is_first=*/false,
                                             /*Check_inf=*/Is_causal ||
                                                 Is_local || !Is_even_MN>(
              acc_s, acc_o, params.scale_softmax_log2);
    // if (cute::thread0()) { print(scores_max); print(scores_sum);
    // print(scores); }

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    flash::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

    cute::cp_async_fence();

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV,
                                                  tKgK(_, _, _, n_block - 1),
                                                  tKsK, tKVcKV, tKVpKV);

      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    mask.template apply_mask</*Causal_mask=*/false>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
    softmax.template softmax_rescale_o</*Is_first=*/false,
                                       /*Check_inf=*/Is_local>(
        acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);
  }

  // Epilogue

  Tensor lse =
      softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
          acc_o, params.scale_softmax);
  // if (cute::thread0()) { print(lse); }

  Tensor sOaccum =
      make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)),
                  typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  using SmemTiledCopyO =
      std::conditional_t<!Split, typename Kernel_traits::SmemCopyAtomO,
                         typename Kernel_traits::SmemCopyAtomOaccum>;
  auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
  auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor rO = flash::convert_type<ElementO>(acc_o);
  Tensor taccOrOaccum =
      smem_thr_copy_Oaccum.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(
      sOaccum);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sOaccum is larger than sQ, so we need to syncthreads here
  // TODO: allocate enough smem for sOaccum
  if constexpr (Split) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_oaccum =
      (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q +
       m_block * kBlockM) *
      params.d_rounded;
  const index_t row_offset_lseaccum =
      (Split || !params.unpadded_lse
           ? ((n_split_idx * params.b + bidb) * params.h + bidh) *
                 params.seqlen_q
           : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)) +
      m_block * kBlockM;

  Tensor gOaccum =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(
                                    Split ? params.oaccum_ptr : params.o_ptr) +
                                (Split ? row_offset_oaccum : row_offset_o)),
                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                  make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
  Tensor gLSEaccum = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr
                                                 : params.softmax_lse_ptr) +
          row_offset_lseaccum),
      Shape<Int<kBlockM>>{}, Stride<_1>{});
  // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n",
  // row_offset_o, bidh, gOaccum.data()); }

  GmemTiledCopyO gmem_tiled_copy_Oaccum;
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(
      sOaccum);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

  __syncthreads();

  Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
  cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(
      size<0>(sOaccum), size<1>(sOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(
      cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_Oaccum, tOrOaccum,
                                     tOgOaccum, tOcO, tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_dropout, bool Is_causal,
          bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
          bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_mixed_attn(const Params &params) {
  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  const bool is_cached_kv = reinterpret_cast<bool *>(params.k_head_mask)[bidh];
  const int kv_head_index = reinterpret_cast<int *>(params.k_head_index)[bidh];

  if (is_cached_kv) {
    flash::compute_mixed_attn_1rowblock_cached_kv_with_gather<
        Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
        Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block,
                                               kv_head_index);
  } else {
    flash::compute_mixed_attn_1rowblock_buffer_kv<
        Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN,
        Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block,
                                               kv_head_index);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split,
          bool Append_KV, typename Params>
inline __device__ void compute_mixed_attn_splitkv(const Params &params) {
  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
  // The block index for the head.
  const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;

  // true for cache_kv with gather
  // false for buffer_kv
  const bool is_cached_kv = reinterpret_cast<bool *>(params.k_head_mask)[bidh];
  const int kv_head_index = reinterpret_cast<int *>(params.k_head_index)[bidh];

  if (is_cached_kv) {
    flash::compute_mixed_attn_1rowblock_splitkv_cached_kv_with_gather<
        Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K,
        Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx,
                                      num_n_splits, kv_head_index);
  } else {
    flash::compute_mixed_attn_1rowblock_splitkv_buffer_kv<
        Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K,
        Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx,
                                      num_n_splits, kv_head_index);
  }
}

}  // namespace flash
}  // namespace kvlib