#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DEVICE_CPU = 0,
    DEVICE_CUDA = 1,
} device;

typedef enum {
    TLA_MEMCPY_HOST_TO_HOST,
    TLA_MEMCPY_HOST_TO_DEVICE,
    TLA_MEMCPY_DEVICE_TO_HOST,
    TLA_MEMCPY_DEVICE_TO_DEVICE,
} TLAMemcpyKind;

void* tla_malloc(device device, size_t bytes);
void tla_free(device device, void* ptr);

int tla_memset_safe(device device, void* ptr, int value, size_t bytes);
int tla_memcpy_safe(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind);

void tla_memset(device device, void* ptr, int value, size_t bytes);
void tla_memcpy(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind);

#ifdef __cplusplus
}
#endif
