#pragma once
#include "tinyla/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_add(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cuda_sub(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cuda_mul(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cuda_div(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cuda_pow(Tensor* out, const Tensor** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif