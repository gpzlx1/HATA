#include <immintrin.h>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "cpu_attn_ops.h"

namespace kvlib {

typedef uint16_t ggml_fp16_t;
typedef double ggml_float;

// F32 AVX512
#define GGML_F32_STEP 64
#define GGML_F32_EPR 16

#define GGML_F32x16 __m512
#define GGML_F32x16_ZERO _mm512_setzero_ps()
#define GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#define GGML_F32x16_LOAD _mm512_loadu_ps
#define GGML_F32x16_STORE _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#define GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32x16_ADD _mm512_add_ps
#define GGML_F32x16_MUL _mm512_mul_ps
#define GGML_F32x16_REDUCE(res, x)                \
  do {                                            \
    int offset = GGML_F32_ARR >> 1;               \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    offset >>= 1;                                 \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    offset >>= 1;                                 \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    res = (ggml_float)_mm512_reduce_add_ps(x[0]); \
  } while (0)

#define GGML_F32_VEC GGML_F32x16
#define GGML_F32_VEC_ZERO GGML_F32x16_ZERO
#define GGML_F32_VEC_SET1 GGML_F32x16_SET1
#define GGML_F32_VEC_LOAD GGML_F32x16_LOAD
#define GGML_F32_VEC_STORE GGML_F32x16_STORE
#define GGML_F32_VEC_FMA GGML_F32x16_FMA
#define GGML_F32_VEC_ADD GGML_F32x16_ADD
#define GGML_F32_VEC_MUL GGML_F32x16_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512
#define GGML_F16_STEP 64
#define GGML_F16_EPR 16

// AVX512 has FP16 extension (AVX512_FP16) but I don't have it on my machine so
// I use FP32 instead

#define GGML_F32Cx16 __m512
#define GGML_F32Cx16_ZERO _mm512_setzero_ps()
#define GGML_F32Cx16_SET1(x) _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in
// AVX512F so F16C guard isn't required
#define GGML_F32Cx16_LOAD(x) \
  _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define GGML_F32Cx16_STORE(x, y) \
  _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#define GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32Cx16_ADD _mm512_add_ps
#define GGML_F32Cx16_MUL _mm512_mul_ps
#define GGML_F32Cx16_REDUCE(res, x)               \
  do {                                            \
    int offset = GGML_F32_ARR >> 1;               \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    offset >>= 1;                                 \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    offset >>= 1;                                 \
    for (int i = 0; i < offset; ++i) {            \
      x[i] = _mm512_add_ps(x[i], x[offset + i]);  \
    }                                             \
    res = (ggml_float)_mm512_reduce_add_ps(x[0]); \
  } while (0)

#define GGML_F16_VEC GGML_F32Cx16
#define GGML_F16_VEC_ZERO GGML_F32Cx16_ZERO
#define GGML_F16_VEC_SET1 GGML_F32Cx16_SET1
#define GGML_F16_VEC_LOAD(p, i) GGML_F32Cx16_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#define GGML_F16_VEC_FMA GGML_F32Cx16_FMA
#define GGML_F16_VEC_ADD GGML_F32Cx16_ADD
#define GGML_F16_VEC_MUL GGML_F32Cx16_MUL

#define GGML_F16_VEC_REDUCE GGML_F32Cx16_REDUCE

#define GGML_F32_ARR (GGML_F32_STEP / GGML_F32_EPR)
#define GGML_F16_ARR (GGML_F16_STEP / GGML_F16_EPR)

#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)

// precomputed f32 table for f16 (256 KB)
// On ARM NEON, it's quicker to directly convert x -> x instead of calling into
// ggml_lookup_fp16_to_fp32, so we define GGML_FP16_TO_FP32 and
// GGML_FP32_TO_FP16 elsewhere for NEON. This is also true for POWER9.
#if !defined(GGML_FP16_TO_FP32)

static float ggml_table_f32_f16[1 << 16];
inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
  uint16_t s;
  memcpy(&s, &f, sizeof(uint16_t));
  return ggml_table_f32_f16[s];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#endif

#if !defined(GGML_FP32_TO_FP16)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE / sizeof(float);

inline void set_ggml_f32_to_f16_table() {
  static bool is_first_call = true;

  if (is_first_call) {
    for (uint32_t i = 0; i < (1 << 16); ++i) {
      union {
        uint16_t u16;
        ggml_fp16_t fp16;
      } u = {static_cast<uint16_t>(i)};
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }

    is_first_call = false;
  }
}

inline void prefetch_4_cacheline(const char *ptr, const int prefetch_level) {
  __builtin_prefetch(ptr, 0, prefetch_level);
  __builtin_prefetch(ptr + 1 * CACHE_LINE_SIZE, 0, prefetch_level);
  __builtin_prefetch(ptr + 2 * CACHE_LINE_SIZE, 0, prefetch_level);
  __builtin_prefetch(ptr + 3 * CACHE_LINE_SIZE, 0, prefetch_level);
}

static void ggml_vec_dot_f16(int n, float *__restrict__ s,
                             const ggml_fp16_t *__restrict__ x,
                             const ggml_fp16_t *__restrict__ y) {
  ggml_float sumf = 0.0;

  const int np = (n & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC sum[GGML_F16_ARR] = {GGML_F16_VEC_ZERO};

  GGML_F16_VEC ax[GGML_F16_ARR];
  GGML_F16_VEC ay[GGML_F16_ARR];

  for (int i = 0; i < np; i += GGML_F16_STEP) {
    for (int j = 0; j < GGML_F16_ARR; j++) {
      ax[j] = GGML_F16_VEC_LOAD(x + i + j * GGML_F16_EPR, j);
      ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);

      sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  GGML_F16_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]));
  }

  *s = sumf;
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t *y,
                                      const float v) {
  const int np = (n & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

  GGML_F16_VEC ay[GGML_F16_ARR];

  for (int i = 0; i < np; i += GGML_F16_STEP) {
    for (int j = 0; j < GGML_F16_ARR; j++) {
      ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);
      ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

      GGML_F16_VEC_STORE(y + i + j * GGML_F16_EPR, ay, j);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) * v);
  }
}

inline static void ggml_vec_scale_f32(const int n, float *y, const float v) {
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] *= v;
  }
}

static void ggml_vec_mad_f16(const int n, ggml_fp16_t *__restrict__ y,
                             const ggml_fp16_t *__restrict__ x, const float v) {
  const int np = (n & ~(GGML_F16_STEP - 1));

  GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

  GGML_F16_VEC ax[GGML_F16_ARR];
  GGML_F16_VEC ay[GGML_F16_ARR];

  for (int i = 0; i < np; i += GGML_F16_STEP) {
    for (int j = 0; j < GGML_F16_ARR; j++) {
      ax[j] = GGML_F16_VEC_LOAD(x + i + j * GGML_F16_EPR, j);
      ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);
      ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

      GGML_F16_VEC_STORE(y + i + j * GGML_F16_EPR, ay, j);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) +
                             GGML_FP16_TO_FP32(x[i]) * v);
  }
}

inline static void ggml_vec_mad_f32(const int n, float *__restrict__ y,
                                    const float *__restrict__ x,
                                    const float v) {
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ax[GGML_F32_ARR];
  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] += x[i] * v;
  }
}

void ggml_fp32_to_fp16_row(const float *x, ggml_fp16_t *y, int64_t n) {
  int64_t i = 0;
#if defined(__F16C__)
  // if (ggml_cpu_has_f16c()) {
  for (; i + 7 < n; i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i);
    __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i *)(y + i), y_vec);
  }
  for (; i + 3 < n; i += 4) {
    __m128 x_vec = _mm_loadu_ps(x + i);
    __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64((__m128i *)(y + i), y_vec);
  }
  //}
#endif
  for (; i < n; i++) {
    y[i] = GGML_FP32_TO_FP16(x[i]);
  }
}

void ggml_compute_forward_flash_attn_ext_by_thread(CPUAttnParams &params,
                                                   const int ith, const int nth,
                                                   uint8_t *wdata) {
  set_ggml_f32_to_f16_table();

  // parallelize by q rows using ggml_vec_dot_f32

  const int nr = params.bsz * params.num_heads * params.seqlen_q;
  const int dr = (nr + nth - 1) / nth;
  const int ir0 = dr * ith;
  const int ir1 = std::min(ir0 + dr, nr);

  // loop over n_batch and n_head
  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    // printf("sparse attn execution debug 1\n");
    const int bid = ir / (params.num_heads * params.seqlen_q);
    const int hid =
        (ir - bid * params.num_heads * params.seqlen_q) / params.seqlen_q;
    const int kv_hid = hid / params.gqa_size;
    const int sid =
        (ir - bid * params.num_heads * params.seqlen_q - hid * params.seqlen_q);

    float S = 0.0f;       // sum
    float M = -INFINITY;  // maximum KQ value
    float *VKQ32 =
        (float *)wdata + ith * (3 * params.head_dim +
                                CACHE_LINE_SIZE_F32);  // FP32 VKQ accumulator
    ggml_fp16_t *VKQ16 =
        (ggml_fp16_t *)(VKQ32 +
                        1 * params
                                .head_dim);  // (temporary) FP16 VKQ accumulator
    ggml_fp16_t *Q_q =
        (ggml_fp16_t *)(VKQ32 +
                        2 * params.head_dim);  // (temporary) buffer for Q
                                               // converted to quantized/FP16
    memset(VKQ16, 0, params.head_dim * sizeof(ggml_fp16_t));
    const float *pq = (float *)params.q_ptr + sid * params.q_seq_stride +
                      hid * params.q_head_stride + bid * params.q_bsz_stride;
    ggml_fp32_to_fp16_row(pq, Q_q, params.head_dim);

    ggml_fp16_t *k_data_base_ptr = (ggml_fp16_t *)params.k_ptr +
                                   kv_hid * params.k_head_stride +
                                   bid * params.k_bsz_stride;
    ggml_fp16_t *v_data_base_ptr = (ggml_fp16_t *)params.v_ptr +
                                   kv_hid * params.v_head_stride +
                                   bid * params.v_bsz_stride;

    // online softmax / attention
    // loop over n_kv and n_head_kv
    // ref: https://arxiv.org/pdf/2112.05682.pdf
    for (int64_t ic = 0; ic < params.seqlen_k; ++ic) {
      const ggml_fp16_t *k_data = k_data_base_ptr + ic * params.k_seq_stride;
      const ggml_fp16_t *v_data = v_data_base_ptr + ic * params.v_seq_stride;

      const int cur_prefetch_level = 3;
      prefetch_4_cacheline((char *)k_data, cur_prefetch_level);
      prefetch_4_cacheline((char *)v_data, cur_prefetch_level);

      float s;  // KQ value
      ggml_vec_dot_f16(params.head_dim, &s, k_data, Q_q);
      s = s * params.scale;  // scale KQ value

      const float Mold = M;
      float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with
                        // this value
      float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

      if (s > M) {
        // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
        M = s;
        ms = expf(Mold - M);
        ggml_vec_scale_f16(params.head_dim, VKQ16, ms);
      } else {
        // no new maximum, ms == 1.0f, vs != 1.0f
        vs = expf(s - M);
      }

      ggml_vec_mad_f16(params.head_dim, VKQ16, v_data, vs);
      S = S * ms + vs;  // scale and increment sum with partial sum
    }

    for (int64_t d = 0; d < params.head_dim; ++d) {
      VKQ32[d] = GGML_FP16_TO_FP32(VKQ16[d]);
    }

    // V /= S
    const float S_inv = 1.0f / S;
    ggml_vec_scale_f32(params.head_dim, VKQ32, S_inv);

    if (params.return_lse) {
      float L = M + logf(S);
      memcpy((float *)params.l_ptr + bid * params.l_bsz_stride +
                 hid * params.l_head_stride + sid * params.l_seq_stride,
             &L, sizeof(float));
    }
    memcpy((float *)params.o_ptr + bid * params.o_bsz_stride +
               hid * params.o_head_stride + sid * params.o_seq_stride,
           VKQ32, params.head_dim * sizeof(float));
  }
}

void ggml_compute_forward_sparse_flash_attn_ext_by_thread(
    CPUSparseAttnParams &params, const int ith, const int nth, uint8_t *wdata) {
  set_ggml_f32_to_f16_table();

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = params.seqlen_q * params.num_heads * params.bsz;
  const int dr = (nr + nth - 1) / nth;
  const int ir0 = dr * ith;
  const int ir1 = std::min(ir0 + dr, nr);

  // loop over n_batch and n_head
  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int bid = ir / (params.num_heads * params.seqlen_q);
    const int hid =
        (ir - bid * params.num_heads * params.seqlen_q) / params.seqlen_q;
    const int kv_hid = hid / params.gqa_size;
    const int sid =
        (ir - bid * params.num_heads * params.seqlen_q - hid * params.seqlen_q);

    float S = 0.0f;       // sum
    float M = -INFINITY;  // maximum KQ value
    float *VKQ32 =
        (float *)wdata + ith * (3 * params.head_dim +
                                CACHE_LINE_SIZE_F32);  // FP32 VKQ accumulator
    ggml_fp16_t *VKQ16 =
        (ggml_fp16_t *)(VKQ32 +
                        1 * params
                                .head_dim);  // (temporary) FP16 VKQ accumulator
    ggml_fp16_t *Q_q =
        (ggml_fp16_t *)(VKQ32 +
                        2 * params.head_dim);  // (temporary) buffer for Q
                                               // converted to quantized/FP16

    memset(VKQ16, 0, params.head_dim * sizeof(ggml_fp16_t));
    const float *pq = (float *)params.q_ptr + sid * params.q_seq_stride +
                      hid * params.q_head_stride + bid * params.q_bsz_stride;
    ggml_fp32_to_fp16_row(pq, Q_q, params.head_dim);

    int32_t *selected_ic_base_ptr = (int32_t *)params.i_ptr +
                                    kv_hid * params.i_head_stride +
                                    bid * params.i_bsz_stride;
    __builtin_prefetch(selected_ic_base_ptr, 0, 3);
    ggml_fp16_t *k_data_base_ptr = (ggml_fp16_t *)params.k_ptr +
                                   kv_hid * params.k_head_stride +
                                   bid * params.k_bsz_stride;
    ggml_fp16_t *v_data_base_ptr = (ggml_fp16_t *)params.v_ptr +
                                   kv_hid * params.k_head_stride +
                                   bid * params.v_bsz_stride;

    const int stage = 6;
    const int total_stage = stage + 1;
    int32_t selected_ics[stage + 1];
    int32_t *selected_ic_ptr;

    // online softmax / attention
    // loop over n_kv and n_head_kv
    // ref: https://arxiv.org/pdf/2112.05682.pdf
    for (int64_t ic = 0; ic < params.seqlen_gather; ++ic) {
      if (ic == 0) {
        for (int k = 0; k < stage; k++) {
          selected_ic_ptr = selected_ic_base_ptr + k * params.i_seq_stride;
          selected_ics[k] = *selected_ic_ptr;
          ggml_fp16_t *next_k_data =
              k_data_base_ptr + selected_ics[k] * params.k_seq_stride;
          const int prefetch_level = 1;
          prefetch_4_cacheline((char *)next_k_data, prefetch_level);
        }
      }

      if (ic < params.seqlen_gather - stage) {
        int32_t prefetch_index = ic + stage;
        selected_ic_ptr =
            selected_ic_base_ptr + prefetch_index * params.i_seq_stride;
        selected_ics[prefetch_index % total_stage] = *selected_ic_ptr;
        ggml_fp16_t *next_k_data =
            k_data_base_ptr +
            selected_ics[prefetch_index % total_stage] * params.k_seq_stride;

        const int prefetch_level = 1;
        prefetch_4_cacheline((char *)next_k_data, prefetch_level);
      }

      int32_t selected_ic = selected_ics[ic % total_stage];

      ggml_fp16_t *k_data = k_data_base_ptr + selected_ic * params.k_seq_stride;
      ggml_fp16_t *v_data = v_data_base_ptr + selected_ic * params.v_seq_stride;

      const int cur_prefetch_level = 3;
      prefetch_4_cacheline((char *)k_data, cur_prefetch_level);
      prefetch_4_cacheline((char *)v_data, cur_prefetch_level);

      float s;  // KQ value
      ggml_vec_dot_f16(params.head_dim, &s, k_data, Q_q);

      if (ic == 0) {
        for (int k = 0; k < stage; k++) {
          ggml_fp16_t *next_v_data =
              v_data_base_ptr + selected_ics[k] * params.v_seq_stride;
          const int prefetch_level = 1;
          prefetch_4_cacheline((char *)next_v_data, prefetch_level);
        }
      }

      if (ic < params.seqlen_gather - stage) {
        int32_t prefetch_index = (ic + stage);
        ggml_fp16_t *next_v_data =
            v_data_base_ptr +
            selected_ics[prefetch_index % total_stage] * params.v_seq_stride;
        const int prefetch_level = 1;
        prefetch_4_cacheline((char *)next_v_data, prefetch_level);
      }

      s = s * params.scale;  // scale KQ value

      const float Mold = M;
      float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with
                        // this value
      float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

      if (s > M) {
        // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
        M = s;
        ms = expf(Mold - M);
        ggml_vec_scale_f16(params.head_dim, VKQ16, ms);
      } else {
        // no new maximum, ms == 1.0f, vs != 1.0f
        vs = expf(s - M);
      }

      ggml_vec_mad_f16(params.head_dim, VKQ16, v_data, vs);
      S = S * ms + vs;  // scale and increment sum with partial sum
    }

    for (int64_t d = 0; d < params.head_dim; ++d) {
      VKQ32[d] = GGML_FP16_TO_FP32(VKQ16[d]);
    }

    // V /= S
    const float S_inv = 1.0f / S;
    ggml_vec_scale_f32(params.head_dim, VKQ32, S_inv);

    if (params.return_lse) {
      float L = M + logf(S);
      memcpy((float *)params.l_ptr + bid * params.l_bsz_stride +
                 hid * params.l_head_stride + sid * params.l_seq_stride,
             &L, sizeof(float));
    }
    memcpy((float *)params.o_ptr + bid * params.o_bsz_stride +
               hid * params.o_head_stride + sid * params.o_seq_stride,
           VKQ32, params.head_dim * sizeof(float));
  }
}

}  // namespace kvlib