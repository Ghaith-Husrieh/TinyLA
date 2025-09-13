#pragma once
#include <stddef.h>

int cpu_mul_scalar(const double** inputs, double* out, size_t numel);
int cpu_mul_vec128(const double** inputs, double* out, size_t numel);
int cpu_mul_vec256(const double** inputs, double* out, size_t numel);