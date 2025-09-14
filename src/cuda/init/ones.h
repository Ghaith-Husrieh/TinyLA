#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_ones_kernel(double* buffer, size_t numel);

#ifdef __cplusplus
}
#endif