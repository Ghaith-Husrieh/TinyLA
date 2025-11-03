#include "memory/tensor_desc.h"
#include "philox_common.hpp"
#include <cstddef>

template <typename Generator> tensor_desc* philox_tensor_cpu(const size_t* shape, size_t ndim, uint64_t seed) {
    tensor_desc* desc = tensor_desc_create(NULL, shape, ndim, DEVICE_CPU, buffer_init_mode::UNINITIALIZED);
    PhiloxState st{seed, 0};
    double out[4];
    size_t i = 0;

    while (i < desc->numel) {
        Generator::generate(st, out);

        for (size_t j = 0; j < 4 && i < desc->numel; ++j, ++i) {
            desc->buffer[i] = out[j];
        }
    }

    return desc;
}

#ifdef TINYLA_CUDA_ENABLED
template <typename Generator> tensor_desc* philox_tensor_gpu(const size_t* shape, size_t ndim, uint64_t seed);
#endif
