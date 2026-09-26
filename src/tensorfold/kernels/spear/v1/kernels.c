/* SuperSpear champion kernels — AVX-512 (+ VNNI int8 GEMV).
 *
 * GELU alg  : 0.997729·x·min(1.002, relu(0.306923·x + 0.501)) − 0.004004
 *             (SPEAR evolved algebraic; gelu_policy_generated.py)
 * GELU fast : 1.010719·relu(x) − 0.057684          (ledger, ×6.49 measured)
 * SiLU alg  : x·(0.501 + 0.587·x / (0.815 + √(1+x²)))   (HoF, ×2.43 modelled)
 * SiLU fast : 1.016356·relu(x) − 0.15849            (ledger, ×5.43 measured)
 * Sigmoid fast: 0.605014·x / (1.24384 + |x|) + 0.5  (ledger, ×1.90 measured)
 */
#include "kernels.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline int pad16(int n) { return (n + 15) & ~15; }
static inline int pad4(int n) { return (n + 3) & ~3; }

int spear_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int spear_has_vnni(void) {
#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

/* ---- activations, AVX-512 ------------------------------------------------ */

static inline __m512 relu_ps(__m512 x) { return _mm512_max_ps(x, _mm512_setzero_ps()); }

static inline __m512 gelu_alg_ps(__m512 x) {
    __m512 t = _mm512_fmadd_ps(_mm512_set1_ps(0.306923f), x, _mm512_set1_ps(0.501f));
    t = relu_ps(t);
    t = _mm512_min_ps(t, _mm512_set1_ps(1.002f));
    return _mm512_fmsub_ps(_mm512_mul_ps(_mm512_set1_ps(0.997729f), x), t,
                           _mm512_set1_ps(0.004004f));
}

static inline __m512 gelu_fast_ps(__m512 x) {
    return _mm512_fmadd_ps(_mm512_set1_ps(1.010719f), relu_ps(x), _mm512_set1_ps(-0.057684f));
}

static inline __m512 silu_alg_ps(__m512 x) {
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 den = _mm512_add_ps(_mm512_set1_ps(0.815f),
                               _mm512_sqrt_ps(_mm512_add_ps(_mm512_set1_ps(1.0f), x2)));
    __m512 inner = _mm512_fmadd_ps(_mm512_set1_ps(0.587f), _mm512_div_ps(x, den),
                                   _mm512_set1_ps(0.501f));
    return _mm512_mul_ps(x, inner);
}

static inline __m512 silu_fast_ps(__m512 x) {
    return _mm512_fmadd_ps(_mm512_set1_ps(1.016356f), relu_ps(x), _mm512_set1_ps(-0.15849f));
}

static inline __m512 sigmoid_fast_ps(__m512 x) {
    __m512 den = _mm512_add_ps(_mm512_set1_ps(1.24384f), _mm512_abs_ps(x));
    return _mm512_fmadd_ps(_mm512_set1_ps(0.605014f), _mm512_div_ps(x, den),
                           _mm512_set1_ps(0.5f));
}

static inline float gelu_exact_1(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}
static inline float silu_exact_1(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid_exact_1(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float gelu_alg_1(float x) {
    float t = 0.306923f * x + 0.501f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.002f) t = 1.002f;
    return 0.997729f * x * t - 0.004004f;
}
static inline float gelu_fast_1(float x) {
    return 1.010719f * (x > 0.0f ? x : 0.0f) - 0.057684f;
}
static inline float silu_alg_1(float x) {
    return x * (0.501f + 0.587f * x / (0.815f + sqrtf(1.0f + x * x)));
}
static inline float silu_fast_1(float x) {
    return 1.016356f * (x > 0.0f ? x : 0.0f) - 0.15849f;
}
static inline float sigmoid_fast_1(float x) {
    return 0.605014f * x / (1.24384f + fabsf(x)) + 0.5f;
}

typedef float (*spear_fn1)(float);

static spear_fn1 scalar_fn(int op) {
    switch (op) {
        case SPEAR_GELU_EXACT: return gelu_exact_1;
        case SPEAR_GELU_ALG: return gelu_alg_1;
        case SPEAR_GELU_FAST: return gelu_fast_1;
        case SPEAR_SILU_EXACT: return silu_exact_1;
        case SPEAR_SILU_ALG: return silu_alg_1;
        case SPEAR_SILU_FAST: return silu_fast_1;
        case SPEAR_SIGMOID_EXACT: return sigmoid_exact_1;
        case SPEAR_SIGMOID_FAST: return sigmoid_fast_1;
        default: return gelu_exact_1;
    }
}

#if defined(__AVX512F__)
static int is_avx_op(int op) {
    return op == SPEAR_GELU_ALG || op == SPEAR_GELU_FAST || op == SPEAR_SILU_ALG
        || op == SPEAR_SILU_FAST || op == SPEAR_SIGMOID_FAST;
}

static inline __m512 avx_fn(__m512 x, int op) {
    switch (op) {
        case SPEAR_GELU_ALG: return gelu_alg_ps(x);
        case SPEAR_GELU_FAST: return gelu_fast_ps(x);
        case SPEAR_SILU_ALG: return silu_alg_ps(x);
        case SPEAR_SILU_FAST: return silu_fast_ps(x);
        case SPEAR_SIGMOID_FAST: return sigmoid_fast_ps(x);
        default: return x;
    }
}
#endif

void spear_act_f32(int op, const float *x, float *y, int n) {
    int i = 0;
#if defined(__AVX512F__)
    if (is_avx_op(op)) {
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(x + i);
            _mm512_storeu_ps(y + i, avx_fn(v, op));
        }
    }
#endif
    spear_fn1 fn = scalar_fn(op);
    for (; i < n; i++) y[i] = fn(x[i]);
}

void spear_swiglu_f32(int silu_op, const float *gate, const float *up, float *y, int n) {
    int i = 0;
#if defined(__AVX512F__)
    if (silu_op == SPEAR_SILU_ALG || silu_op == SPEAR_SILU_FAST) {
        for (; i + 16 <= n; i += 16) {
            __m512 g = avx_fn(_mm512_loadu_ps(gate + i), silu_op);
            _mm512_storeu_ps(y + i, _mm512_mul_ps(g, _mm512_loadu_ps(up + i)));
        }
    }
#endif
    spear_fn1 fn = scalar_fn(silu_op);
    for (; i < n; i++) y[i] = fn(gate[i]) * up[i];
}

/* ---- fp32 GEMV / GEMM ---------------------------------------------------- */

void spear_gemv_f32(const float *W, const float *x, const float *bias, float *y, int N, int K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        const float *row = W + (size_t)n * (size_t)K;
        int k = 0;
        float s = 0.0f;
#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        for (; k + 16 <= K; k += 16) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(row + k), _mm512_loadu_ps(x + k), acc);
        }
        s = _mm512_reduce_add_ps(acc);
#endif
        for (; k < K; k++) s += row[k] * x[k];
        if (bias) s += bias[n];
        y[n] = s;
    }
}

void spear_gemm_f32(const float *A, const float *W, const float *bias, float *C,
                    int M, int N, int K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++) {
        const float *a = A + (size_t)m * (size_t)K;
        float *c = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            const float *row = W + (size_t)n * (size_t)K;
            int k = 0;
            float s = 0.0f;
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            for (; k + 16 <= K; k += 16) {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(row + k), _mm512_loadu_ps(a + k), acc);
            }
            s = _mm512_reduce_add_ps(acc);
#endif
            for (; k < K; k++) s += row[k] * a[k];
            if (bias) s += bias[n];
            c[n] = s;
        }
    }
}

/* ---- int8 VNNI GEMV ------------------------------------------------------ */

void spear_quantize_weight_i8(const float *W, int8_t *Wq, float *scale, int32_t *col_sum,
                              int N, int K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        const float *row = W + (size_t)n * (size_t)K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        scale[n] = s;
        int32_t sum = 0;
        int8_t *qrow = Wq + (size_t)n * (size_t)K;
        for (int k = 0; k < K; k++) {
            int v = (int)lrintf(row[k] / s);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qrow[k] = (int8_t)v;
            sum += v;
        }
        col_sum[n] = sum;
    }
}

int spear_pack_bytes(int N, int K) {
    return pad16(N) / 16 * (pad4(K) / 4) * 64;
}

void spear_pack_i8(const int8_t *Wq, int8_t *packed, int N, int K) {
    int Np = pad16(N);
    int Kp = pad4(K);
    int nblocks = Np / 16;
    int kblocks = Kp / 4;
    memset(packed, 0, (size_t)nblocks * (size_t)kblocks * 64u);
    for (int n = 0; n < N; n++) {
        int nb = n / 16;
        int lane = n % 16;
        const int8_t *row = Wq + (size_t)n * (size_t)K;
        for (int k = 0; k < K; k++) {
            packed[(size_t)nb * kblocks * 64 + (size_t)(k / 4) * 64 + (size_t)lane * 4 + (k % 4)] = row[k];
        }
    }
}

static void gemv_i8_body(const int8_t *packed, const float *scale, const int32_t *col_sum,
                         const float *x, const float *bias, float *y, int N, int K, int act_op) {
    int Np = pad16(N);
    int Kp = pad4(K);
    int nblocks = Np / 16;
    int kblocks = Kp / 4;

    float amax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(x[k]);
        if (a > amax) amax = a;
    }
    float x_scale = amax / 127.0f;
    if (x_scale < 1e-12f) x_scale = 1e-12f;

    size_t xq_bytes = ((size_t)Kp + 63u) & ~63u;
    static _Thread_local uint8_t *xq_buf = NULL;
    static _Thread_local size_t xq_cap = 0;
    if (xq_cap < xq_bytes) {
        free(xq_buf);
        xq_buf = (uint8_t *)aligned_alloc(64, xq_bytes);
        xq_cap = xq_buf ? xq_bytes : 0;
    }
    uint8_t *xq = xq_buf;
    if (!xq) return;
    memset(xq, 128, xq_bytes);
    for (int k = 0; k < K; k++) {
        int v = (int)lrintf(x[k] / x_scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        xq[k] = (uint8_t)(v + 128);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int nb = 0; nb < nblocks; nb++) {
        const int8_t *block = packed + (size_t)nb * (size_t)kblocks * 64u;
#if defined(__AVX512VNNI__)
        __m512i acc = _mm512_setzero_si512();
        for (int k4 = 0; k4 < kblocks; k4++) {
            uint32_t four;
            memcpy(&four, xq + k4 * 4, 4);
            __m512i a = _mm512_set1_epi32((int)four);
            __m512i w = _mm512_loadu_si512((const void *)(block + k4 * 64));
            acc = _mm512_dpbusd_epi32(acc, a, w);
        }
        int32_t accs[16];
        _mm512_storeu_si512((void *)accs, acc);
#else
        int32_t accs[16] = {0};
        for (int k4 = 0; k4 < kblocks; k4++) {
            const int8_t *w = block + k4 * 64;
            const uint8_t *a = xq + k4 * 4;
            for (int lane = 0; lane < 16; lane++) {
                const int8_t *wl = w + lane * 4;
                accs[lane] += (int32_t)a[0] * wl[0] + (int32_t)a[1] * wl[1]
                            + (int32_t)a[2] * wl[2] + (int32_t)a[3] * wl[3];
            }
        }
#endif
        spear_fn1 fn = (act_op >= 0) ? scalar_fn(act_op) : NULL;
        for (int i = 0; i < 16; i++) {
            int n = nb * 16 + i;
            if (n >= N) break;
            float v = (float)(accs[i] - 128 * col_sum[n]) * x_scale * scale[n];
            if (bias) v += bias[n];
            y[n] = fn ? fn(v) : v;
        }
    }
}

void spear_gemv_i8(const int8_t *packed, const float *scale, const int32_t *col_sum,
                   const float *x, const float *bias, float *y, int N, int K) {
    gemv_i8_body(packed, scale, col_sum, x, bias, y, N, K, -1);
}

void spear_gemv_i8_act(const int8_t *packed, const float *scale, const int32_t *col_sum,
                       const float *x, const float *bias, float *y, int N, int K, int act_op) {
    gemv_i8_body(packed, scale, col_sum, x, bias, y, N, K, act_op);
}
