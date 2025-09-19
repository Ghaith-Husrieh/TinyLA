#pragma once
#include "tinyla/tensor.h"
#include <malloc.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t batch_dim;
    size_t m;
    size_t k;
    size_t n;
} GemmDims;

static inline GemmDims extract_gemm_dims(const Tensor* a, const Tensor* b) {
    GemmDims dims;
    dims.batch_dim = (a->ndim < 3) ? 0 : 1;
    for (size_t i = 0; i < a->ndim - 2; i++) {
        dims.batch_dim *= a->shape[i];
    }
    dims.m = a->shape[a->ndim - 2];
    dims.k = a->shape[a->ndim - 1];
    dims.n = b->shape[b->ndim - 1];
    return dims;
}

static inline double* pack_matrix(const double* b, size_t k, size_t n) {
    double* b_packed = (double*)malloc(k * n * sizeof(double));
    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        for (size_t j = 0; j < n; j++) {
            b_packed[j * k + k_idx] = b[k_idx * n + j];
        }
    }
    return b_packed;
}

static inline double* pack_batch_matrices(const double* b, size_t batch_dim, size_t k, size_t n) {
    double* b_packed = (double*)malloc(batch_dim * k * n * sizeof(double));
    for (size_t batch = 0; batch < batch_dim; batch++) {
        const size_t b_offset = batch * k * n;
        const size_t b_packed_offset = batch * k * n;
        for (size_t k_idx = 0; k_idx < k; k_idx++) {
            for (size_t j = 0; j < n; j++) {
                b_packed[b_packed_offset + j * k + k_idx] = b[b_offset + k_idx * n + j];
            }
        }
    }
    return b_packed;
}

#ifdef __cplusplus
}
#endif
