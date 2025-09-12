#include "../add.h"
#include <immintrin.h>

int cpu_add_vec128(const double** inputs, double* out, size_t numel) {
    size_t i = 0;
    size_t simd_width = 2;

    for (; i + simd_width <= numel; i += simd_width) {
        __m128d a = _mm_loadu_pd(&inputs[0][i]);
        __m128d b = _mm_loadu_pd(&inputs[1][i]);
        __m128d c = _mm_add_pd(a, b);
        _mm_storeu_pd(&out[i], c);
    }

    for (; i < numel; i++) {
        out[i] = inputs[0][i] + inputs[1][i];
    }

    return 0;
}