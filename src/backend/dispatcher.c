#include "dispatcher.h"
#include "../cpu/cpu_features.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DeviceKernel select_cpu_kernel(const CpuKernels* kernels) {
    if (kernels->vec256 && has_avx2()) {
        return kernels->vec256;
    }

    if (kernels->vec128 && has_sse42()) {
        return kernels->vec128;
    }

    if (kernels->scalar) {
        return kernels->scalar;
    }

    return NULL;
}

DeviceKernel select_gpu_kernel(DeviceKernel gpu_kernel) {
#ifdef TINYLA_CUDA_ENABLED
    return gpu_kernel;
#else
    return NULL;
#endif
}

OpEntry* get_op_entry(OpType op) {
    if (op < 0 || op >= OP_COUNT) {
        return NULL;
    }
    return &op_table[op];
}

int register_op(const char* verbose_name,
                OpType op,
                OpKind kind,
                OpArity arity,
                OpValidator validator,
                DeviceKernel cpu_k,
                DeviceKernel gpu_k) {
    if (op < 0 || op >= OP_COUNT) {
        fprintf(stderr, "Invalid op type: %d\n", op);
        return -1;
    }
    if (arity < OP_ARITY_UNARY || arity > OP_ARITY_TERNARY) {
        fprintf(stderr, "Invalid op arity: %d\n", arity);
        return -1;
    }

    op_table[op].verbose_name = verbose_name;
    op_table[op].kind = kind;
    op_table[op].arity = arity;
    op_table[op].validator = validator;
    op_table[op].cpu_kernel = cpu_k;
#ifdef TINYLA_CUDA_ENABLED
    op_table[op].gpu_kernel = gpu_k;
#endif
    return 0;
}

int dispatch_kernel(const OpEntry* entry, const tensor_desc** inputs, const size_t n_inputs, tensor_desc* out) {
    DeviceKernel kernel = NULL;
    if (out->device == DEVICE_CPU) {
        kernel = entry->cpu_kernel;
    }
#ifdef TINYLA_CUDA_ENABLED
    else if (out->device == DEVICE_CUDA) {
        kernel = entry->gpu_kernel;
    }
#endif
    else {
        fprintf(stderr, "Unsupported device %d\n", out->device);
        return -1;
    }

    if (!kernel) {
        fprintf(stderr, "No kernel registered for %s on device %d\n", entry->verbose_name, out->device);
        return -1;
    }

    return kernel(out, inputs, n_inputs);
}
