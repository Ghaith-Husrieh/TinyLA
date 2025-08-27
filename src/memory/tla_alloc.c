#include "tla_alloc.h"
#include <stdlib.h>
#include <string.h>

#ifdef TINYLA_CUDA_ENABLED
#include "../cuda/cuda_macros.h"
#include <cuda_runtime.h>
#endif

void* tla_malloc(Device device, size_t bytes) {
#ifdef TINYLA_CUDA_ENABLED
    size_t required_bytes = bytes == 0 ? 1 : bytes;
    if (device == DEVICE_GPU) {
        void* ptr = NULL;
        if (cudaMalloc(&ptr, required_bytes) != cudaSuccess)
            return NULL;
        return ptr;
    }
#endif
    return malloc(required_bytes);
}

void tla_free(Device device, void* ptr) {
    if (!ptr)
        return;

#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU) {
        cudaFree(ptr);
        return;
    }
#endif
    free(ptr);
}

int tla_memset_safe(Device device, void* ptr, int value, size_t bytes) {
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU) {
        return cudaMemset(ptr, value, bytes) == cudaSuccess ? 0 : -1;
    }
#endif
    memset(ptr, value, bytes);
    return 0;
}

int tla_memcpy_safe(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind) {
    switch (kind) {
    case TLA_MEMCPY_HOST_TO_HOST:
        memcpy(dst, src, bytes);
        return 0;
#ifdef TINYLA_CUDA_ENABLED
    case TLA_MEMCPY_HOST_TO_DEVICE:
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
    case TLA_MEMCPY_DEVICE_TO_HOST:
        return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
    case TLA_MEMCPY_DEVICE_TO_DEVICE:
        return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess ? 0 : -1;
#endif
    default:
        if (kind == TLA_MEMCPY_HOST_TO_DEVICE || kind == TLA_MEMCPY_DEVICE_TO_HOST ||
            kind == TLA_MEMCPY_DEVICE_TO_DEVICE) {
            fprintf(stderr, "TinyLA was compiled without CUDA support\n");
            return -1;
        } else {
            fprintf(stderr, "Invalid memcpy kind: %d\n", kind);
            return -1;
        }
    }
}

void tla_memset(Device device, void* ptr, int value, size_t bytes) {
#ifdef TINYLA_CUDA_ENABLED
    if (device == DEVICE_GPU) {
        TLA_CUDA_CHECK(cudaMemset(ptr, value, bytes));
        return;
    }
#endif
    memset(ptr, value, bytes);
}

void tla_memcpy(void* dst, const void* src, size_t bytes, TLAMemcpyKind kind) {
    switch (kind) {
    case TLA_MEMCPY_HOST_TO_HOST:
        memcpy(dst, src, bytes);
        break;
#ifdef TINYLA_CUDA_ENABLED
    case TLA_MEMCPY_HOST_TO_DEVICE:
        TLA_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        break;
    case TLA_MEMCPY_DEVICE_TO_HOST:
        TLA_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        break;
    case TLA_MEMCPY_DEVICE_TO_DEVICE:
        TLA_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
        break;
#endif
    default:
        if (kind == TLA_MEMCPY_HOST_TO_DEVICE || kind == TLA_MEMCPY_DEVICE_TO_HOST ||
            kind == TLA_MEMCPY_DEVICE_TO_DEVICE) {
            fprintf(stderr, "TinyLA was compiled without CUDA support\n");
            break;
        } else {
            fprintf(stderr, "Invalid memcpy kind: %d\n", kind);
            break;
        }
    }
}