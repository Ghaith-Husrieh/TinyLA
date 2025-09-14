#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int gpu_add_kernel(const double** inputs, double* out, size_t numel);

#ifdef __cplusplus
}
#endif