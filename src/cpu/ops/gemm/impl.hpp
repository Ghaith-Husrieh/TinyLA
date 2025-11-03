#pragma once
#include "common.hpp"
#include "functors.hpp"
#include <omp.h>

template <typename Op> void gemm_single_impl(const double* a, const double* b, double* out, const GemmDims* dims) {
    double* b_packed = pack_matrix(b, dims->k, dims->n);

    int m = (int)dims->m;
    int n = (int)dims->n;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Op::apply(a, b_packed, out, i, j, dims->k, dims->n);
        }
    }

    free(b_packed);
}

template <typename Op> void gemm_batch_large_impl(const double* a, const double* b, double* out, const GemmDims* dims) {
    double* b_packed = pack_batch_matrices(b, dims->batch_dim, dims->k, dims->n);

    int batch_dim = (int)dims->batch_dim;
    int m = (int)dims->m;
    int n = (int)dims->n;

#pragma omp parallel for
    for (int batch = 0; batch < batch_dim; batch++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Op::apply_batch(a, b_packed, out, batch, i, j, dims->m, dims->k, dims->n);
            }
        }
    }

    free(b_packed);
}

template <typename Op> void gemm_batch_small_impl(const double* a, const double* b, double* out, const GemmDims* dims) {
    double* b_packed = pack_batch_matrices(b, dims->batch_dim, dims->k, dims->n);

    int batch_dim = (int)dims->batch_dim;
    int m = (int)dims->m;
    int n = (int)dims->n;

#pragma omp parallel for collapse(3)
    for (int batch = 0; batch < batch_dim; batch++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Op::apply_batch(a, b_packed, out, batch, i, j, dims->m, dims->k, dims->n);
            }
        }
    }

    free(b_packed);
}
