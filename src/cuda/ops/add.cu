#include "../cuda_macros.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_kernel(const double* a, const double* b, double* out, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" int gpu_add_kernel(double** inputs, double* out, size_t numel) {
    if (!inputs || !out) {
        fprintf(stderr, "gpu_add_kernel: null pointer\n");
        return -1;
    }

    const double* a = (const double*)inputs[0];
    const double* b = (const double*)inputs[1];
    if (!a || !b) {
        fprintf(stderr, "gpu_add_kernel: null input buffer\n");
        return -1;
    }

    if (numel == 0) {
        return 0;
    }

    size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, numel);
    TLA_CUDA_KERNEL_CHECK();

    return 0;
}