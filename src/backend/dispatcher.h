#pragma once
#include "tinyla/tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    // Binary Operations
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,

    // Operation Count
    OP_COUNT,
} OpType;

typedef enum {
    OP_ARITY_UNARY = 1,
    OP_ARITY_BINARY = 2,
    OP_ARITY_TERNARY = 3,
} OpArity;

typedef int (*DeviceKernel)(const double** inputs, double* out, size_t numel);

typedef struct {
    DeviceKernel scalar;
    DeviceKernel vec128;
    DeviceKernel vec256;
} CpuKernels;

typedef struct {
    OpArity arity;
    DeviceKernel cpu_kernel;
#ifdef TINYLA_CUDA_ENABLED
    DeviceKernel gpu_kernel;
#endif
} OpEntry;

static OpEntry op_table[OP_COUNT];

DeviceKernel select_cpu_kernel(const CpuKernels* kernels);
DeviceKernel select_gpu_kernel(DeviceKernel gpu_kernel);

int register_op(OpType op, OpArity arity, DeviceKernel cpu_k, DeviceKernel gpu_k);
int dispatch_op(OpType op, Tensor* out, const Tensor** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif