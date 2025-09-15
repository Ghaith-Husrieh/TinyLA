#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Add operations
int cpu_add_scalar(const double** inputs, double* out, size_t numel);
int cpu_add_vec128(const double** inputs, double* out, size_t numel);
int cpu_add_vec256(const double** inputs, double* out, size_t numel);

// Sub operations
int cpu_sub_scalar(const double** inputs, double* out, size_t numel);
int cpu_sub_vec128(const double** inputs, double* out, size_t numel);
int cpu_sub_vec256(const double** inputs, double* out, size_t numel);

// Mul operations
int cpu_mul_scalar(const double** inputs, double* out, size_t numel);
int cpu_mul_vec128(const double** inputs, double* out, size_t numel);
int cpu_mul_vec256(const double** inputs, double* out, size_t numel);

// Div operations
int cpu_div_scalar(const double** inputs, double* out, size_t numel);
int cpu_div_vec128(const double** inputs, double* out, size_t numel);
int cpu_div_vec256(const double** inputs, double* out, size_t numel);

#ifdef __cplusplus
}
#endif