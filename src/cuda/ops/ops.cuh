#pragma once
#include "../cuda_macros.h"
#include "functors.cuh"
#include <cuda_runtime.h>

template <typename Op> __global__ void binary_op_kernel(const double* a, const double* b, double* out, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = Op::apply(a[idx], b[idx]);
    }
}

template <typename Op> int cuda_binary_op_impl(const double* a, const double* b, double* out, size_t numel) {
    const size_t block_size = 256;
    const size_t grid_size = (numel + block_size - 1) / block_size;

    binary_op_kernel<Op><<<grid_size, block_size>>>(a, b, out, numel);
    TLA_CUDA_KERNEL_CHECK();

    return 0;
}