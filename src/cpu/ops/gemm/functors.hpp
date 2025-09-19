#pragma once
#include <immintrin.h>

struct GemmScalarOp {
    static void apply(const double* a, const double* b_packed, double* out, size_t i, size_t j, size_t k, size_t n) {
        double sum = 0.0;

        size_t k_idx = 0;
        for (; k_idx + 3 < k; k_idx += 4) {
            sum += a[i * k + k_idx] * b_packed[j * k + k_idx];
            sum += a[i * k + k_idx + 1] * b_packed[j * k + k_idx + 1];
            sum += a[i * k + k_idx + 2] * b_packed[j * k + k_idx + 2];
            sum += a[i * k + k_idx + 3] * b_packed[j * k + k_idx + 3];
        }
        for (; k_idx < k; k_idx++) {
            sum += a[i * k + k_idx] * b_packed[j * k + k_idx];
        }
        out[i * n + j] = sum;
    }

    static void apply_batch(const double* a, const double* b_packed, double* out, size_t batch, size_t i, size_t j,
                            size_t m, size_t k, size_t n) {
        const size_t a_offset = batch * m * k;
        const size_t b_packed_offset = batch * k * n;
        const size_t c_offset = batch * m * n;

        double sum = 0.0;

        size_t k_idx = 0;
        for (; k_idx + 3 < k; k_idx += 4) {
            sum += a[a_offset + i * k + k_idx] * b_packed[b_packed_offset + j * k + k_idx];
            sum += a[a_offset + i * k + k_idx + 1] * b_packed[b_packed_offset + j * k + k_idx + 1];
            sum += a[a_offset + i * k + k_idx + 2] * b_packed[b_packed_offset + j * k + k_idx + 2];
            sum += a[a_offset + i * k + k_idx + 3] * b_packed[b_packed_offset + j * k + k_idx + 3];
        }
        for (; k_idx < k; k_idx++) {
            sum += a[a_offset + i * k + k_idx] * b_packed[b_packed_offset + j * k + k_idx];
        }
        out[c_offset + i * n + j] = sum;
    }
};

struct GemmVec128Op {
    static void apply(const double* a, const double* b_packed, double* out, size_t i, size_t j, size_t k, size_t n) {
        __m128d sum = _mm_setzero_pd();

        size_t k_idx = 0;
        for (; k_idx + 1 < k; k_idx += 2) {
            __m128d a_vec = _mm_loadu_pd(&a[i * k + k_idx]);
            __m128d b_vec = _mm_loadu_pd(&b_packed[j * k + k_idx]);
            sum = _mm_add_pd(sum, _mm_mul_pd(a_vec, b_vec));
        }
        for (; k_idx < k; k_idx++) {
            sum = _mm_add_pd(sum, _mm_mul_pd(_mm_set1_pd(a[i * k + k_idx]), _mm_set1_pd(b_packed[j * k + k_idx])));
        }

        __m128d sum_64 = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));

        out[i * n + j] = _mm_cvtsd_f64(sum_64);
    }

    static void apply_batch(const double* a, const double* b_packed, double* out, size_t batch, size_t i, size_t j,
                            size_t m, size_t k, size_t n) {
        const size_t a_offset = batch * m * k;
        const size_t b_packed_offset = batch * k * n;
        const size_t c_offset = batch * m * n;

        __m128d sum = _mm_setzero_pd();

        size_t k_idx = 0;
        for (; k_idx + 1 < k; k_idx += 2) {
            __m128d a_vec = _mm_loadu_pd(&a[a_offset + i * k + k_idx]);
            __m128d b_vec = _mm_loadu_pd(&b_packed[b_packed_offset + j * k + k_idx]);
            sum = _mm_add_pd(sum, _mm_mul_pd(a_vec, b_vec));
        }
        for (; k_idx < k; k_idx++) {
            sum = _mm_add_pd(sum, _mm_mul_pd(_mm_set1_pd(a[a_offset + i * k + k_idx]),
                                             _mm_set1_pd(b_packed[b_packed_offset + j * k + k_idx])));
        }

        __m128d sum_64 = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));

        out[c_offset + i * n + j] = _mm_cvtsd_f64(sum_64);
    }
};

struct GemmVec256Op {
    static void apply(const double* a, const double* b_packed, double* out, size_t i, size_t j, size_t k, size_t n) {
        __m256d sum = _mm256_setzero_pd();

        size_t k_idx = 0;
        for (; k_idx + 3 < k; k_idx += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i * k + k_idx]);
            __m256d b_vec = _mm256_loadu_pd(&b_packed[j * k + k_idx]);
            sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
        }
        for (; k_idx < k; k_idx++) {
            sum = _mm256_fmadd_pd(_mm256_set1_pd(a[i * k + k_idx]), _mm256_set1_pd(b_packed[j * k + k_idx]), sum);
        }

        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_128 = _mm_add_pd(sum_high, sum_low);
        __m128d sum_64 = _mm_add_pd(sum_128, _mm_unpackhi_pd(sum_128, sum_128));

        out[i * n + j] = _mm_cvtsd_f64(sum_64);
    }

    static void apply_batch(const double* a, const double* b_packed, double* out, size_t batch, size_t i, size_t j,
                            size_t m, size_t k, size_t n) {
        const size_t a_offset = batch * m * k;
        const size_t b_packed_offset = batch * k * n;
        const size_t c_offset = batch * m * n;

        __m256d sum = _mm256_setzero_pd();

        size_t k_idx = 0;
        for (; k_idx + 3 < k; k_idx += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[a_offset + i * k + k_idx]);
            __m256d b_vec = _mm256_loadu_pd(&b_packed[b_packed_offset + j * k + k_idx]);
            sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
        }
        for (; k_idx < k; k_idx++) {
            sum = _mm256_fmadd_pd(_mm256_set1_pd(a[a_offset + i * k + k_idx]),
                                  _mm256_set1_pd(b_packed[b_packed_offset + j * k + k_idx]), sum);
        }

        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_128 = _mm_add_pd(sum_high, sum_low);
        __m128d sum_64 = _mm_add_pd(sum_128, _mm_unpackhi_pd(sum_128, sum_128));

        out[c_offset + i * n + j] = _mm_cvtsd_f64(sum_64);
    }
};
