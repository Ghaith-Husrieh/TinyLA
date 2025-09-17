#pragma once
#include "tinyla/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Add operations
int cpu_add_scalar(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_add_vec128(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_add_vec256(Tensor* out, const Tensor** inputs, size_t n_inputs);

// Sub operations
int cpu_sub_scalar(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_sub_vec128(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_sub_vec256(Tensor* out, const Tensor** inputs, size_t n_inputs);

// Mul operations
int cpu_mul_scalar(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_mul_vec128(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_mul_vec256(Tensor* out, const Tensor** inputs, size_t n_inputs);

// Div operations
int cpu_div_scalar(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_div_vec128(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_div_vec256(Tensor* out, const Tensor** inputs, size_t n_inputs);

// Pow operations
int cpu_pow_scalar(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_pow_vec128(Tensor* out, const Tensor** inputs, size_t n_inputs);
int cpu_pow_vec256(Tensor* out, const Tensor** inputs, size_t n_inputs);

#ifdef __cplusplus
}
#endif