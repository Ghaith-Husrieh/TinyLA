#pragma once
#include "../memory/tensor_desc.h"
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
    OP_POW,
    OP_MATMUL,

    // Operation Count
    OP_COUNT,
} OpType;

typedef enum {
    OP_ELEMENT_WISE,
    OP_GEMM,

} OpKind;

typedef enum {
    OP_ARITY_UNARY = 1,
    OP_ARITY_BINARY = 2,
    OP_ARITY_TERNARY = 3,
} OpArity;

typedef int (*DeviceKernel)(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
typedef int (*OpValidator)(const tensor_desc** inputs, size_t n_inputs);

typedef struct {
    const char* verbose_name;
    OpKind kind;
    OpArity arity;
    OpValidator validator;

    DeviceKernel cpu_kernel;
#ifdef TINYLA_CUDA_ENABLED
    DeviceKernel gpu_kernel;
#endif
} OpEntry;

static OpEntry op_table[OP_COUNT];
OpEntry* get_op_entry(OpType op);

typedef struct {
    DeviceKernel scalar;
    DeviceKernel vec128;
    DeviceKernel vec256;
} CpuKernels;

DeviceKernel select_cpu_kernel(const CpuKernels* kernels);
DeviceKernel select_gpu_kernel(DeviceKernel gpu_kernel);

int register_op(const char* verbose_name,
                OpType op,
                OpKind kind,
                OpArity arity,
                OpValidator validator,
                DeviceKernel cpu_k,
                DeviceKernel gpu_k);
int dispatch_kernel(const OpEntry* entry, const tensor_desc** inputs, const size_t n_inputs, tensor_desc* out);

#ifdef __cplusplus
}
#endif
