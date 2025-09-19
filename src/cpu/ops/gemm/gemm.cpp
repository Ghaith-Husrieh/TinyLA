#include "gemm.h"
#include "functors.hpp"
#include "impl.hpp"
#include <omp.h>

int cpu_gemm_scalar(Tensor* out, const Tensor** inputs, const size_t n_inputs) {
    const Tensor* a = inputs[0];
    const Tensor* b = inputs[1];

    GemmDims dims = extract_gemm_dims(a, b);

    if (dims.batch_dim == 0) {
        gemm_single_impl<GemmScalarOp>(a->buffer, b->buffer, out->buffer, &dims);
    } else if (dims.batch_dim >= omp_get_max_threads()) {
        gemm_batch_large_impl<GemmScalarOp>(a->buffer, b->buffer, out->buffer, &dims);
    } else {
        gemm_batch_small_impl<GemmScalarOp>(a->buffer, b->buffer, out->buffer, &dims);
    }

    return 0;
}

int cpu_gemm_vec128(Tensor* out, const Tensor** inputs, const size_t n_inputs) {
    const Tensor* a = inputs[0];
    const Tensor* b = inputs[1];

    GemmDims dims = extract_gemm_dims(a, b);

    if (dims.batch_dim == 0) {
        gemm_single_impl<GemmVec128Op>(a->buffer, b->buffer, out->buffer, &dims);
    } else if (dims.batch_dim >= omp_get_max_threads()) {
        gemm_batch_large_impl<GemmVec128Op>(a->buffer, b->buffer, out->buffer, &dims);
    } else {
        gemm_batch_small_impl<GemmVec128Op>(a->buffer, b->buffer, out->buffer, &dims);
    }

    return 0;
}

int cpu_gemm_vec256(Tensor* out, const Tensor** inputs, const size_t n_inputs) {
    const Tensor* a = inputs[0];
    const Tensor* b = inputs[1];

    GemmDims dims = extract_gemm_dims(a, b);

    if (dims.batch_dim == 0) {
        gemm_single_impl<GemmVec256Op>(a->buffer, b->buffer, out->buffer, &dims);
    } else if (dims.batch_dim >= omp_get_max_threads()) {
        gemm_batch_large_impl<GemmVec256Op>(a->buffer, b->buffer, out->buffer, &dims);
    } else {
        gemm_batch_small_impl<GemmVec256Op>(a->buffer, b->buffer, out->buffer, &dims);
    }

    return 0;
}