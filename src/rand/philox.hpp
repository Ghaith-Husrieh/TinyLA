#include "philox_common.hpp"
#include "tinyla/tensor.h"
#include <cstddef>

template <typename Generator> Tensor* philox_tensor_cpu(const size_t* shape, size_t ndim, uint64_t seed) {
    Tensor* tensor = empty_tensor(shape, ndim, DEVICE_CPU);
    PhiloxState st{seed, 0};
    double out[4];
    size_t i = 0;

    while (i < tensor->numel) {
        Generator::generate(st, out);

        for (size_t j = 0; j < 4 && i < tensor->numel; ++j, ++i) {
            tensor->buffer[i] = out[j];
        }
    }

    return tensor;
}

#ifdef TINYLA_CUDA_ENABLED
template <typename Generator> Tensor* philox_tensor_gpu(const size_t* shape, size_t ndim, uint64_t seed);
#endif