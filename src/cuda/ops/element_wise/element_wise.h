#pragma once
#include "../../../memory/tensor_desc.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_add(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cuda_sub(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cuda_mul(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cuda_div(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cuda_pow(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif