#pragma once
#include "../../../memory/tensor_desc.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Add operations
int cpu_add_scalar(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_add_vec128(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_add_vec256(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);

// Sub operations
int cpu_sub_scalar(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_sub_vec128(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_sub_vec256(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);

// Mul operations
int cpu_mul_scalar(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_mul_vec128(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_mul_vec256(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);

// Div operations
int cpu_div_scalar(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_div_vec128(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_div_vec256(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);

// Pow operations
int cpu_pow_scalar(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_pow_vec128(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);
int cpu_pow_vec256(tensor_desc* out, const tensor_desc** inputs, size_t n_inputs);

#ifdef __cplusplus
}
#endif