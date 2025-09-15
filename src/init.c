#include "tinyla/init.h"
#include "backend/dispatcher.h"
#include "cpu/ops/element_wise.h"

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/ops/add.h"
#include "cuda/ops/div.h"
#include "cuda/ops/mul.h"
#include "cuda/ops/sub.h"
#endif

void tinyla_init(void) {
    // === Add Kernels ===
    CpuKernels add_kernels = {
            .scalar = cpu_add_scalar,
            .vec128 = cpu_add_vec128,
            .vec256 = cpu_add_vec256,
    };
    register_op(OP_ADD, OP_ARITY_BINARY, select_cpu_kernel(&add_kernels), select_gpu_kernel(gpu_add_kernel));

    // === Sub Kernels ===
    CpuKernels sub_kernels = {
            .scalar = cpu_sub_scalar,
            .vec128 = cpu_sub_vec128,
            .vec256 = cpu_sub_vec256,
    };
    register_op(OP_SUB, OP_ARITY_BINARY, select_cpu_kernel(&sub_kernels), select_gpu_kernel(gpu_sub_kernel));

    // === Mul Kernels ===
    CpuKernels mul_kernels = {
            .scalar = cpu_mul_scalar,
            .vec128 = cpu_mul_vec128,
            .vec256 = cpu_mul_vec256,
    };
    register_op(OP_MUL, OP_ARITY_BINARY, select_cpu_kernel(&mul_kernels), select_gpu_kernel(gpu_mul_kernel));

    // === Div Kernels ===
    CpuKernels div_kernels = {
            .scalar = cpu_div_scalar,
            .vec128 = cpu_div_vec128,
            .vec256 = cpu_div_vec256,
    };
    register_op(OP_DIV, OP_ARITY_BINARY, select_cpu_kernel(&div_kernels), select_gpu_kernel(gpu_div_kernel));
}