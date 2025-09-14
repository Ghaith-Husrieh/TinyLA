#include "tinyla/rand.h"
#include "philox.hpp"

Tensor* rand_tensor(const size_t* shape, size_t ndim, Device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxUniform>(shape, ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU)
        return philox_tensor_gpu<PhiloxUniform>(shape, ndim, seed);
#endif
    return nullptr;
}

Tensor* randn_tensor(const size_t* shape, size_t ndim, Device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxNormal>(shape, ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU)
        return philox_tensor_gpu<PhiloxNormal>(shape, ndim, seed);
#endif
    return nullptr;
}

Tensor* rand_tensor_like(const Tensor* tensor, Device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxUniform>(tensor->shape, tensor->ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU)
        return philox_tensor_gpu<PhiloxUniform>(tensor->shape, tensor->ndim, seed);
#endif
    return nullptr;
}

Tensor* randn_tensor_like(const Tensor* tensor, Device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxNormal>(tensor->shape, tensor->ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU)
        return philox_tensor_gpu<PhiloxNormal>(tensor->shape, tensor->ndim, seed);
#endif
    return nullptr;
}