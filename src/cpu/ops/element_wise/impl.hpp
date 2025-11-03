#pragma once
#include "../../../memory/tensor_desc.h"
#include "functors.hpp"
#include <immintrin.h>
#include <omp.h>

template <typename Op>
int cpu_element_wise_scalar_impl(tensor_desc* out, const tensor_desc* a, const tensor_desc* b, const size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i++) {
        out->buffer[i] = Op::apply(a->buffer[i], b->buffer[i]);
    }

    return 0;
}

template <typename Op>
int cpu_element_wise_vec128_impl(tensor_desc* out, const tensor_desc* a, const tensor_desc* b, const size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i += 2) {
        if (i + 1 < (int)numel) {
            __m128d va = _mm_load_pd(&a->buffer[i]);
            __m128d vb = _mm_load_pd(&b->buffer[i]);
            __m128d result = Op::apply_simd(va, vb);
            _mm_store_pd(&out->buffer[i], result);
        } else {
            out->buffer[i] = Op::apply(a->buffer[i], b->buffer[i]);
        }
    }

    return 0;
}

template <typename Op>
int cpu_element_wise_vec256_impl(tensor_desc* out, const tensor_desc* a, const tensor_desc* b, const size_t numel) {
#pragma omp parallel for
    for (int i = 0; i < (int)numel; i += 4) {
        if (i + 3 < (int)numel) {
            __m256d va = _mm256_load_pd(&a->buffer[i]);
            __m256d vb = _mm256_load_pd(&b->buffer[i]);
            __m256d result = Op::apply_simd256(va, vb);
            _mm256_store_pd(&out->buffer[i], result);
        } else {
            for (int j = i; j < (int)numel; j++) {
                out->buffer[j] = Op::apply(a->buffer[j], b->buffer[j]);
            }
        }
    }

    return 0;
}
