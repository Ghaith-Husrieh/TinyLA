#include "rand/rand.h"
#include "memory/tensor_desc.h"
#include "philox.hpp"

tensor_desc* rand_tensor(const size_t* shape, size_t ndim, device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxUniform>(shape, ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_CUDA)
        return philox_tensor_gpu<PhiloxUniform>(shape, ndim, seed);
#endif
    return nullptr;
}

tensor_desc* randn_tensor(const size_t* shape, size_t ndim, device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxNormal>(shape, ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_CUDA)
        return philox_tensor_gpu<PhiloxNormal>(shape, ndim, seed);
#endif
    return nullptr;
}

tensor_desc* rand_tensor_like(const tensor_desc* tensor, device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxUniform>(tensor->shape, tensor->ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_CUDA)
        return philox_tensor_gpu<PhiloxUniform>(tensor->shape, tensor->ndim, seed);
#endif
    return nullptr;
}

tensor_desc* randn_tensor_like(const tensor_desc* tensor, device device, uint64_t seed) {
    if (device == DEVICE_CPU)
        return philox_tensor_cpu<PhiloxNormal>(tensor->shape, tensor->ndim, seed);
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_CUDA)
        return philox_tensor_gpu<PhiloxNormal>(tensor->shape, tensor->ndim, seed);
#endif
    return nullptr;
}
