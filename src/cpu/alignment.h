#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define __tla_aligned__(x) __declspec(align(x))
#else
#define __tla_aligned__(x) __attribute__((aligned(x)))
#endif

bool is_aligned(void* ptr, size_t alignment);
void* tla_aligned_malloc(size_t bytes, size_t alignment);
void tla_aligned_free(void* ptr);

#ifdef __cplusplus
}
#endif
