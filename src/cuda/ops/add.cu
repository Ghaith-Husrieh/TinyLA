#include "../cuda_macros.h"
#include <cuda_runtime.h>

__global__ void add_kernel(const double* a, const double* b, double* out, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" int gpu_add_kernel(const double** inputs, double* out, size_t numel) {
    size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(inputs[0], inputs[1], out, numel);
    TLA_CUDA_KERNEL_CHECK();

    return 0;
}