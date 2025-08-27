#pragma once
#include "tinyla/tensor.h"
#include <stddef.h>

typedef enum {
    TLA_MEMCPY_HOST_TO_HOST,
    TLA_MEMCPY_HOST_TO_DEVICE,
    TLA_MEMCPY_DEVICE_TO_HOST,
    TLA_MEMCPY_DEVICE_TO_DEVICE,
} TLAMemcpyKind;

void* tla_malloc(Device device, size_t bytes);
void tla_free(Device device, void* ptr);

int tla_memset_safe(Device device, void* ptr, int value, size_t bytes);
int tla_memcpy_safe(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind);

void tla_memset(Device device, void* ptr, int value, size_t bytes);
void tla_memcpy(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind);
