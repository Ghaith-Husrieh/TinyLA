#include "../sub.h"
#include <immintrin.h>
#include <omp.h>

int cpu_sub_vec128(const double** inputs, double* out, size_t numel) {
    size_t simd_width = 2;
    size_t n_simd = numel / simd_width;
    size_t tail_start = n_simd * simd_width;
    int i;

#pragma omp parallel for
    for (i = 0; i < n_simd; i++) {
        size_t idx = i * simd_width;
        __m128d a = _mm_loadu_pd(&inputs[0][idx]);
        __m128d b = _mm_loadu_pd(&inputs[1][idx]);
        __m128d c = _mm_sub_pd(a, b);
        _mm_storeu_pd(&out[idx], c);
    }

    for (size_t i = tail_start; i < numel; i++) {
        out[i] = inputs[0][i] - inputs[1][i];
    }

    return 0;
}