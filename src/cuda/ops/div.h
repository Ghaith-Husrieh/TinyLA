#pragma once
#include <stddef.h>

int gpu_div_kernel(const double** inputs, double* out, size_t numel);