#pragma once
#include "../../cuda_macros.h"
#include "functors.cuh"
#include <cuda_runtime.h>

template <typename Op>
__global__ void element_wise_kernel(double* out, const double* a, const double* b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = Op::apply(a[idx], b[idx]);
    }
}

template <typename Op>
int cuda_element_wise_impl(tensor_desc* out, const tensor_desc* a, const tensor_desc* b, const size_t numel) {
    const size_t block_size = 256;
    const size_t grid_size = (numel + block_size - 1) / block_size;

    element_wise_kernel<Op><<<grid_size, block_size>>>(out->buffer, a->buffer, b->buffer, numel);
    TLA_CUDA_KERNEL_CHECK();

    return 0;
}
