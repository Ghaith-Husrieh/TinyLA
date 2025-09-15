#pragma once
#include <cmath>
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

struct PowOp {
    static double apply(double a, double b) {
        if (b == 0.0)
            return 1.0;
        if (b == 1.0)
            return a;
        if (b == -1.0)
            return 1.0 / a;
        if (b == 2.0)
            return a * a;
        if (b == 3.0)
            return a * a * a;
        if (b == -2.0)
            return 1.0 / (a * a);
        return std::pow(a, b);
    }

    static __m128d apply_simd(__m128d a, __m128d b) {
        __m128d b_rounded = _mm_round_pd(b, _MM_FROUND_TO_NEAREST_INT);
        __m128d diff = _mm_sub_pd(b, b_rounded);
        __m128d abs_diff = _mm_andnot_pd(_mm_set1_pd(-0.0), diff);
        __m128d is_integer = _mm_cmplt_pd(abs_diff, _mm_set1_pd(1e-10));

        if (_mm_movemask_pd(is_integer) == 0x3) {
            __m128d b_int = _mm_round_pd(b, _MM_FROUND_TO_NEAREST_INT);
            __m128d is_zero = _mm_cmpeq_pd(b_int, _mm_set1_pd(0.0));
            __m128d is_one = _mm_cmpeq_pd(b_int, _mm_set1_pd(1.0));
            __m128d is_neg_one = _mm_cmpeq_pd(b_int, _mm_set1_pd(-1.0));
            __m128d is_two = _mm_cmpeq_pd(b_int, _mm_set1_pd(2.0));
            __m128d is_three = _mm_cmpeq_pd(b_int, _mm_set1_pd(3.0));
            __m128d is_neg_two = _mm_cmpeq_pd(b_int, _mm_set1_pd(-2.0));

            if (_mm_movemask_pd(is_zero) == 0x3) {
                return _mm_set1_pd(1.0);
            } else if (_mm_movemask_pd(is_one) == 0x3) {
                return a;
            } else if (_mm_movemask_pd(is_neg_one) == 0x3) {
                return _mm_div_pd(_mm_set1_pd(1.0), a);
            } else if (_mm_movemask_pd(is_two) == 0x3) {
                return _mm_mul_pd(a, a);
            } else if (_mm_movemask_pd(is_three) == 0x3) {
                __m128d a_squared = _mm_mul_pd(a, a);
                return _mm_mul_pd(a_squared, a);
            } else if (_mm_movemask_pd(is_neg_two) == 0x3) {
                __m128d a_squared = _mm_mul_pd(a, a);
                return _mm_div_pd(_mm_set1_pd(1.0), a_squared);
            }
        }

        double result[2];
        for (int i = 0; i < 2; i++) {
            result[i] = apply(((double*)&a)[i], ((double*)&b)[i]);
        }
        return _mm_load_pd(result);
    }

    static __m256d apply_simd256(__m256d a, __m256d b) {
        __m256d b_rounded = _mm256_round_pd(b, _MM_FROUND_TO_NEAREST_INT);
        __m256d diff = _mm256_sub_pd(b, b_rounded);
        __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);
        __m256d is_integer = _mm256_cmp_pd(abs_diff, _mm256_set1_pd(1e-10), _CMP_LT_OQ);

        if (_mm256_movemask_pd(is_integer) == 0xF) {
            __m256d b_int = _mm256_round_pd(b, _MM_FROUND_TO_NEAREST_INT);
            __m256d is_zero = _mm256_cmp_pd(b_int, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
            __m256d is_one = _mm256_cmp_pd(b_int, _mm256_set1_pd(1.0), _CMP_EQ_OQ);
            __m256d is_neg_one = _mm256_cmp_pd(b_int, _mm256_set1_pd(-1.0), _CMP_EQ_OQ);
            __m256d is_two = _mm256_cmp_pd(b_int, _mm256_set1_pd(2.0), _CMP_EQ_OQ);
            __m256d is_three = _mm256_cmp_pd(b_int, _mm256_set1_pd(3.0), _CMP_EQ_OQ);
            __m256d is_neg_two = _mm256_cmp_pd(b_int, _mm256_set1_pd(-2.0), _CMP_EQ_OQ);

            if (_mm256_movemask_pd(is_zero) == 0xF) {
                return _mm256_set1_pd(1.0);
            } else if (_mm256_movemask_pd(is_one) == 0xF) {
                return a;
            } else if (_mm256_movemask_pd(is_neg_one) == 0xF) {
                return _mm256_div_pd(_mm256_set1_pd(1.0), a);
            } else if (_mm256_movemask_pd(is_two) == 0xF) {
                return _mm256_mul_pd(a, a);
            } else if (_mm256_movemask_pd(is_three) == 0xF) {
                __m256d a_squared = _mm256_mul_pd(a, a);
                return _mm256_mul_pd(a_squared, a);
            } else if (_mm256_movemask_pd(is_neg_two) == 0xF) {
                __m256d a_squared = _mm256_mul_pd(a, a);
                return _mm256_div_pd(_mm256_set1_pd(1.0), a_squared);
            }
        }

        double result[4];
        for (int i = 0; i < 4; i++) {
            result[i] = apply(((double*)&a)[i], ((double*)&b)[i]);
        }
        return _mm256_load_pd(result);
    }
};