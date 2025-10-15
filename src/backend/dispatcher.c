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

int register_op(OpType op, OpArity arity, DeviceKernel cpu_k, DeviceKernel gpu_k) {
    if (op < 0 || op >= OP_COUNT) {
        fprintf(stderr, "Invalid op type: %d\n", op);
        return -1;
    }
    if (arity < OP_ARITY_UNARY || arity > OP_ARITY_TERNARY) {
        fprintf(stderr, "Invalid op arity: %d\n", arity);
        return -1;
    }

    op_table[op].arity = arity;
    op_table[op].cpu_kernel = cpu_k;
#ifdef TINYLA_CUDA_ENABLED
    op_table[op].gpu_kernel = gpu_k;
#endif
    return 0;
}

int dispatch_op(OpType op, tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    if (op < 0 || op >= OP_COUNT) {
        fprintf(stderr, "Invalid op type: %d\n", op);
        return -1;
    }

    OpEntry* entry = &op_table[op];
    if (entry->arity != n_inputs) {
        fprintf(stderr, "Op %d expects %d inputs, but got %zu\n", op, entry->arity, n_inputs);
        return -1;
    }

    DeviceKernel kernel = NULL;
    if (inputs[0]->device == DEVICE_CPU) {
        kernel = entry->cpu_kernel;
    }
#ifdef TINYLA_CUDA_ENABLED
    else if (inputs[0]->device == DEVICE_CUDA) {
        kernel = entry->gpu_kernel;
    }
#endif
    else {
        fprintf(stderr, "Unsupported device %d\n", inputs[0]->device);
        return -1;
    }

    if (!kernel) {
        fprintf(stderr, "No kernel registered for op %d on device %d\n", op, inputs[0]->device);
        return -1;
    }

    int result = kernel(out, inputs, n_inputs);

    return result;
}