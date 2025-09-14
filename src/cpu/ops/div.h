#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cpu_div_scalar(const double** inputs, double* out, size_t numel);
int cpu_div_vec128(const double** inputs, double* out, size_t numel);
int cpu_div_vec256(const double** inputs, double* out, size_t numel);

#ifdef __cplusplus
}
#endif