#include "../cuda_macros.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ones_kernel(double* buffer, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        buffer[idx] = 1.0;
    }
}

extern "C" void launch_ones_kernel(double* buffer, size_t numel) {
    size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    ones_kernel<<<blocks, threads>>>(buffer, numel);
    TLA_CUDA_KERNEL_CHECK();
}
