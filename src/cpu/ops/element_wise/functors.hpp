#pragma once
#include <immintrin.h>

struct AddOp {
    static double apply(double a, double b) { return a + b; }
    static __m128d apply_simd(__m128d a, __m128d b) { return _mm_add_pd(a, b); }
    static __m256d apply_simd256(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
};

struct SubOp {
    static double apply(double a, double b) { return a - b; }
    static __m128d apply_simd(__m128d a, __m128d b) { return _mm_sub_pd(a, b); }
    static __m256d apply_simd256(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
};

struct MulOp {
    static double apply(double a, double b) { return a * b; }
    static __m128d apply_simd(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
    static __m256d apply_simd256(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
};

struct DivOp {
    static double apply(double a, double b) { return a / b; }
    static __m128d apply_simd(__m128d a, __m128d b) { return _mm_div_pd(a, b); }
    static __m256d apply_simd256(__m256d a, __m256d b) { return _mm256_div_pd(a, b); }
};