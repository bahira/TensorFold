/* SuperSpear champion kernels for TensorFold (CPU, AVX-512).
 *
 * Formulas from https://github.com/bahira/superspear (MIT), ledger 2026-08.
 * Three operating points per activation:
 *   exact  — libm transcendental (the serial reference)
 *   alg    — evolved algebraic, no exp/erf/tanh
 *   fast   — SuperSpear fast slot (relu / rational, measured 1.9–6.5×)
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum spear_op {
    SPEAR_GELU_EXACT = 0,
    SPEAR_GELU_ALG = 1,
    SPEAR_GELU_FAST = 2,
    SPEAR_SILU_EXACT = 3,
    SPEAR_SILU_ALG = 4,
    SPEAR_SILU_FAST = 5,
    SPEAR_SIGMOID_EXACT = 6,
    SPEAR_SIGMOID_FAST = 7,
};

void spear_act_f32(int op, const float *x, float *y, int n);
void spear_swiglu_f32(int silu_op, const float *gate, const float *up, float *y, int n);

void spear_gemv_f32(const float *W, const float *x, const float *bias, float *y, int N, int K);
void spear_gemm_f32(const float *A, const float *W, const float *bias, float *C,
                    int M, int N, int K);

void spear_quantize_weight_i8(const float *W, int8_t *Wq, float *scale, int32_t *col_sum,
                              int N, int K);
int spear_pack_bytes(int N, int K);
void spear_pack_i8(const int8_t *Wq, int8_t *packed, int N, int K);
void spear_gemv_i8(const int8_t *packed, const float *scale, const int32_t *col_sum,
                   const float *x, const float *bias, float *y, int N, int K);
void spear_gemv_i8_act(const int8_t *packed, const float *scale, const int32_t *col_sum,
                       const float *x, const float *bias, float *y, int N, int K, int act_op);

int spear_has_avx512(void);
int spear_has_vnni(void);

#ifdef __cplusplus
}
#endif
