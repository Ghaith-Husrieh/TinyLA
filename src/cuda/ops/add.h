#pragma once
#include <stddef.h>

int gpu_add_kernel(const double** inputs, double* out, size_t numel);