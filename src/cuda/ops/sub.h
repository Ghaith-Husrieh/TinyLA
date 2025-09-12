#pragma once
#include <stddef.h>

int gpu_sub_kernel(const double** inputs, double* out, size_t numel);