#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "cute/tensor.hpp"
#include "operator.h"

#define PRINT(name, content) \
  print(name);               \
  print(" : ");              \
  print(content);            \
  print("\n");

#define PRINTTENSOR(name, content) \
  print(name);                     \
  print(" : ");                    \
  print_tensor(content);           \
  print("\n");

namespace kvlib {

using namespace cute;
using T = cute::half_t;

template <typename Config>
__global__ void /* __launch_bounds__(128, 1) */
gemm_multi_stage(int32_t *Pack_Dptr, void *Dptr, const void *Aptr,
                 const void *Bptr, int m, int n, int k) {
  using namespace cute;
  using X = Underscore;

  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;

  constexpr int WarpSize = 32;
  int lane_id = idx % WarpSize;
  int warp_id = idx / WarpSize;
  int num_warp = blockDim.x / WarpSize;

  int ix = blockIdx.x;
  int iy = blockIdx.y;
  bool m_check_boundary = m < (iy + 1) * kTileM;

  // use Tensor notation to represent device pointer + dimension
  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));  // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));  // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));  // (M, N)
  Tensor PD =
      make_tensor(make_gmem_ptr((int32_t *)Pack_Dptr), make_shape(m, n / 32),
                  make_stride(n / 32, Int<1>{}));  // (M, N)

  // slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _));  // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _));  // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix));  // (kTileM, kTileN)
  Tensor gPD = local_tile(PD, make_tile(Int<kTileM>{}, Int<kTileN / 32>{}),
                          make_coord(iy, ix));  // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{});  // (kTileN, kTileK, kStage)

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  // method
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

  // fill zero for accumulator
  clear(tCrD);

  // gmem -cp.async-> shm -ldmatrix-> reg
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

  auto cA = make_identity_tensor(
      make_shape(size<0>(sA), size<1>(sA)));   // (kTileM, kTileK)
  auto tAcA = g2s_thr_copy_a.partition_S(cA);  // (CPY, CPY_N, CPY_K)

  auto cB = make_identity_tensor(
      make_shape(size<0>(sB), size<1>(sB)));   // (kTileN, kTileK)
  auto tBcB = g2s_thr_copy_b.partition_S(cB);  // (CPY, CPY_N, CPY_K)

  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  // for predict
  Tensor tApA = make_tensor<bool>(
      make_shape(size<1>(tAsA_copy), size<2>(tAsA_copy)), Stride<_1, _0>{});
  auto m_max_coord = m - size<0>(gA) * iy;

  if (m_check_boundary) {
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor tAcA = g2s_thr_copy_a.partition_S(cA);
#pragma unroll
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m, 0) =
          get<0>(tAcA(0, m, 0)) < m_max_coord;  // blk_m coord < residue_m
    }
  }

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (m_check_boundary) {
      clear(tAsA_copy(_, _, _, istage));
      cute::copy_if(g2s_tiled_copy_a, tApA, tAgA_copy(_, _, _, istage),
                    tAsA_copy(_, _, _, istage));
    } else {
      cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                 tAsA_copy(_, _, _, istage));
    }

    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));

    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // smem -> reg
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          if (m_check_boundary) {
            clear(tAsA_copy(_, _, _, ismem_write));
            cute::copy_if(g2s_tiled_copy_a, tApA,
                          tAgA_copy(_, _, _, itile_to_read),
                          tAsA_copy(_, _, _, ismem_write));
          } else {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                       tAsA_copy(_, _, _, ismem_write));
          }

          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }  // for ik
  }  // itile
  __syncthreads();

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

  auto cC = make_identity_tensor(make_shape(size<0>(gD), size<1>(gD)));
  auto tCcC = s2g_thr_copy_c.partition_S(cC);  // (CPY, CPY_M, CPY_M)
  auto tCcC_group = group_modes<1, 3>(tCcC);   // (CPY_, CPY_MN)

  int step = size<3>(tCsC_r2s);  // pipe

  int sc_m = size<0>(sC);
  int sc_n = size<1>(sC);
  int sc_k = size<2>(sC);

  // if (threadIdx.x == 0){
  //   printf("sc_m: %d, sc_n: %d, sc_k: %d\n", sc_m, sc_n, sc_k);
  // }

  assert(sc_n % 32 == 0);

  int _32_tile_m = kTileM / sc_m;
  int _32_tile_n = kTileN / sc_n;

  // if (threadIdx.x == 0) {
  //   PRINTTENSOR("tCsC_s2g", tCsC_s2g);
  // }

#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // we add a temp tensor to cope with accumulator and output data type
      // difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < step; j++) {
      int _smem_chunk_id = i + j;
      int m_offset = (_smem_chunk_id % _32_tile_m) * sc_m;
      int n_offset = (_smem_chunk_id / _32_tile_m) * sc_n;
      int end_k = min(sc_m, m_max_coord - m_offset);
      int z = _smem_chunk_id / _32_tile_m;

      for (int k = warp_id; k < end_k; k += num_warp) {
        for (int l = lane_id; l < sc_n; l += WarpSize) {
          // gD(m_offset + k, n_offset + l) = sC(k, l, j);
          // a += (half)(sC(k, l, j));
          auto value = sC(k, l, j);
          int _data = value > 0;
          int packbit = __ballot_sync(0xffffffff, _data);
          // int packbit = 0;
          //
          // if (threadIdx.x == 0) {
          //   printf("%d\n", packbit);
          // }
          if (lane_id == 0) {
            gPD(m_offset + k, z) = packbit;
          }
        }
      }
    }

    // #pragma unroll
    //     for (int j = 0; j < step; ++j) {
    //       // if (threadIdx.x == 0) {
    //       //   printf("%f\n", float(sC[0]));
    //       // }
    //       gD[0] = sC[0];

    //       // for (int k = warp_id; k < sc_m; k += num_warp){

    //       // }
    //     }

    // #pragma unroll
    //   for(int j = warp_id; j < sc_m * sc_k; j += num_warp) {
    //     for(int k = lane_id; k < sc_n; k += lane_size) {
    //       gD(sc_m * sc_k * i + j, k) = sC(j / sc_k, k);
    //     }

    //   }

    // #pragma unroll
    //     // shm -> global
    //     for (int j = 0; j < step; ++j) {
    //       // can only check boundary with the granularity of S2GCopyAtomC
    //       bool this_atom_copy = true;

    //       if (m_check_boundary && get<0>(tCcC_group(0, i + j)) >= m - iy *
    //       kTileM) {
    //         this_atom_copy = false;
    //       }

    //       if (this_atom_copy) {
    //         cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i
    //         + j));
    //       }
    //     }

    __syncthreads();
  }
}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  // using mma_op = SM80_16x8x16_F32F16F16F32_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                      make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{});
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}  // namespace config

torch::Tensor EncodeCUDA(torch::Tensor data, torch::Tensor hash_weight) {
  // shape for data is (BATCH, #HEAD, HEAD_DIM)
  // shape for hash_weight is (HEAD_DIM, RBIT) # avoid transpose

  CHECK(data.is_cuda() && data.is_contiguous());
  CHECK(hash_weight.is_cuda() && hash_weight.is_contiguous());

  // int32_t batch_size = data.size(0);
  // int32_t num_head = data.size(1);
  // int32_t head_dim = data.size(2);
  // int32_t rbit = hash_weight.size(0);

  // CHECK(head_dim % 32 == 0);
  // CHECK(rbit % 32 == 0);

  // int32_t total_num_head = batch_size * num_head;

  int32_t M = data.size(0);
  int32_t K = data.size(1);
  int32_t N = hash_weight.size(0);
  CHECK(N % 32 == 0);
  CHECK(K % 32 == 0);

  auto device = data.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor output = torch::empty({M, N}, data.options());
  torch::Tensor packbits_output = torch::empty(
      {M, N / 32}, torch::TensorOptions().device(device).dtype(torch::kInt32));

  if (N >= 128) {
    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int shm_size = gemm_config.kShmSize;

    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)><<<grid, block, shm_size>>>(
        packbits_output.data_ptr<int32_t>(), output.data_ptr(), data.data_ptr(),
        hash_weight.data_ptr(), M, N, K);
  } else if (N == 64) {
    config::GemmConfig<T, 128, 64, 32, 3> gemm_config;

    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int shm_size = gemm_config.kShmSize;

    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)><<<grid, block, shm_size>>>(
        packbits_output.data_ptr<int32_t>(), output.data_ptr(), data.data_ptr(),
        hash_weight.data_ptr(), M, N, K);
  } else if (N == 32) {
    config::GemmConfig<T, 128, 32, 32, 3> gemm_config;

    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int shm_size = gemm_config.kShmSize;

    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    gemm_multi_stage<decltype(gemm_config)><<<grid, block, shm_size>>>(
        packbits_output.data_ptr<int32_t>(), output.data_ptr(), data.data_ptr(),
        hash_weight.data_ptr(), M, N, K);
  }

  // return output;
  return packbits_output;
}

}  // namespace kvlib