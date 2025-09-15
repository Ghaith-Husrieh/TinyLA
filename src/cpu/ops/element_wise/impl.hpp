#pragma once
#include "functors.hpp"
#include <immintrin.h>

template <typename Op> int cpu_element_wise_scalar_impl(const double** inputs, double* out, size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i++) {
        out[i] = Op::apply(inputs[0][i], inputs[1][i]);
    }

    return 0;
}

template <typename Op> int cpu_element_wise_vec128_impl(const double** inputs, double* out, size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i += 2) {
        if (i + 1 < (int)numel) {
            __m128d a = _mm_load_pd(&inputs[0][i]);
            __m128d b = _mm_load_pd(&inputs[1][i]);
            __m128d result = Op::apply_simd(a, b);
            _mm_store_pd(&out[i], result);
        } else {
            out[i] = Op::apply(inputs[0][i], inputs[1][i]);
        }
    }

    return 0;
}

template <typename Op> int cpu_element_wise_vec256_impl(const double** inputs, double* out, size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i += 4) {
        if (i + 3 < (int)numel) {
            __m256d a = _mm256_load_pd(&inputs[0][i]);
            __m256d b = _mm256_load_pd(&inputs[1][i]);
            __m256d result = Op::apply_simd256(a, b);
            _mm256_store_pd(&out[i], result);
        } else {
            for (int j = i; j < (int)numel; j++) {
                out[j] = Op::apply(inputs[0][j], inputs[1][j]);
            }
        }
    }

    return 0;
}