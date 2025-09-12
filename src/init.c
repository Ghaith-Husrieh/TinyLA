#include "tinyla/init.h"
#include "backend/dispatcher.h"
#include "cpu/ops/add.h"

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/ops/add.h"
#endif

void tinyla_init(void) {
    // === Add Kernels ===
    CpuKernels add_kernels = {
            .scalar = cpu_add_scalar,
            .vec128 = cpu_add_vec128,
            .vec256 = cpu_add_vec256,
    };
    register_op(OP_ADD, OP_ARITY_BINARY, select_cpu_kernel(&add_kernels), select_gpu_kernel(gpu_add_kernel));
}