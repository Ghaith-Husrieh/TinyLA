#include "../cuda/cuda_macros.h"
#include "philox.hpp"
#include "philox_common.hpp"
#include "tinyla/tensor.h"
#include <cuda_runtime.h>

template <typename Generator> __global__ void philox_kernel(double* buffer, size_t numel, uint64_t seed) {
    size_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base_idx >= numel)
        return;

    PhiloxState st{seed, 0};
    double out[4];

    Generator::generate(st, out, base_idx);

    for (int i = 0; i < 4; ++i) {
        if (base_idx + i < numel)
            buffer[base_idx + i] = out[i];
    }
}

template <typename Generator> Tensor* philox_tensor_gpu(const size_t* shape, size_t ndim, uint64_t seed) {
    Tensor* tensor = empty_tensor(shape, ndim, DEVICE_GPU);

    size_t threads = 256;
    size_t blocks = (tensor->numel + 4 * threads - 1) / (4 * threads);

    philox_kernel<Generator><<<blocks, threads>>>(tensor->buffer, tensor->numel, seed);
    TLA_CUDA_KERNEL_CHECK();

    return tensor;
}

template __global__ void philox_kernel<PhiloxUniform>(double* buffer, size_t numel, uint64_t seed);
template __global__ void philox_kernel<PhiloxNormal>(double* buffer, size_t numel, uint64_t seed);
template Tensor* philox_tensor_gpu<PhiloxUniform>(const size_t* shape, size_t ndim, uint64_t seed);
template Tensor* philox_tensor_gpu<PhiloxNormal>(const size_t* shape, size_t ndim, uint64_t seed);
