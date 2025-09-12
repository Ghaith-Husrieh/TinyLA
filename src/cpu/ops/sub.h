#pragma once
#include <stddef.h>

int cpu_sub_scalar(const double** inputs, double* out, size_t numel);
int cpu_sub_vec128(const double** inputs, double* out, size_t numel);
int cpu_sub_vec256(const double** inputs, double* out, size_t numel);