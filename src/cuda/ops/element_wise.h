#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_add(const double** inputs, double* out, size_t numel);
int cuda_sub(const double** inputs, double* out, size_t numel);
int cuda_mul(const double** inputs, double* out, size_t numel);
int cuda_div(const double** inputs, double* out, size_t numel);

#ifdef __cplusplus
}
#endif