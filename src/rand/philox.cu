#include "../cuda/cuda_macros.h"
#include "memory/tensor_desc.h"
#include "philox.hpp"
#include "philox_common.hpp"
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

template <typename Generator> tensor_desc* philox_tensor_gpu(const size_t* shape, size_t ndim, uint64_t seed) {
    tensor_desc* desc = tensor_desc_create(NULL, shape, ndim, DEVICE_CUDA, buffer_init_mode::UNINITIALIZED);

    size_t threads = 256;
    size_t blocks = (desc->numel + 4 * threads - 1) / (4 * threads);

    philox_kernel<Generator><<<blocks, threads>>>(desc->buffer, desc->numel, seed);
    TLA_CUDA_KERNEL_CHECK();

    return desc;
}

template __global__ void philox_kernel<PhiloxUniform>(double* buffer, size_t numel, uint64_t seed);
template __global__ void philox_kernel<PhiloxNormal>(double* buffer, size_t numel, uint64_t seed);
template tensor_desc* philox_tensor_gpu<PhiloxUniform>(const size_t* shape, size_t ndim, uint64_t seed);
template tensor_desc* philox_tensor_gpu<PhiloxNormal>(const size_t* shape, size_t ndim, uint64_t seed);
