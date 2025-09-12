#pragma once
#include <stddef.h>

int cpu_add_scalar(const double** inputs, double* out, size_t numel);
int cpu_add_vec128(const double** inputs, double* out, size_t numel);
int cpu_add_vec256(const double** inputs, double* out, size_t numel);