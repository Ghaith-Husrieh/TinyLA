#pragma once
#include "memory/tensor_desc.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

tensor_desc* rand_tensor(const size_t* shape, size_t ndim, device device, uint64_t seed);
tensor_desc* randn_tensor(const size_t* shape, size_t ndim, device device, uint64_t seed);
tensor_desc* rand_tensor_like(const tensor_desc* tensor, device device, uint64_t seed);
tensor_desc* randn_tensor_like(const tensor_desc* tensor, device device, uint64_t seed);

#ifdef __cplusplus
}
#endif
