#include "../add.h"
#include <immintrin.h>

int cpu_add_vec256(const double** inputs, double* out, size_t numel) {
    size_t i = 0;
    size_t simd_width = 4;

    for (; i + simd_width <= numel; i += simd_width) {
        __m256d a = _mm256_loadu_pd(&inputs[0][i]);
        __m256d b = _mm256_loadu_pd(&inputs[1][i]);
        __m256d c = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&out[i], c);
    }

    for (; i < numel; i++) {
        out[i] = inputs[0][i] + inputs[1][i];
    }

    return 0;
}