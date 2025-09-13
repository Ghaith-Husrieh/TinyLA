#pragma once
#include <stddef.h>

int gpu_mul_kernel(const double** inputs, double* out, size_t numel);