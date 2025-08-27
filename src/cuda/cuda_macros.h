#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TLA_CUDA_CHECK(call)                                                                                           \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define TLA_CUDA_KERNEL_CHECK()                                                                                        \
    do {                                                                                                               \
        cudaError_t err = cudaGetLastError();                                                                          \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "[CUDA KERNEL ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)