#include "tinyla/init.h"
#include "backend/dispatcher.h"
#include "backend/validators.h"
#include "cpu/ops/element_wise/element_wise.h"
#include "cpu/ops/gemm/gemm.h"

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/ops/element_wise/element_wise.h"
#include "cuda/ops/gemm/gemm.h"
#endif

void tinyla_init(void) {
    // === Add Kernels ===
    CpuKernels add_kernels = {
            .scalar = cpu_add_scalar,
            .vec128 = cpu_add_vec128,
            .vec256 = cpu_add_vec256,
    };
    register_op("Add",
                OP_ADD,
                OP_ELEMENT_WISE,
                OP_ARITY_BINARY,
                validate_binary_element_wise,
                select_cpu_kernel(&add_kernels),
                select_gpu_kernel(cuda_add));

    // === Sub Kernels ===
    CpuKernels sub_kernels = {
            .scalar = cpu_sub_scalar,
            .vec128 = cpu_sub_vec128,
            .vec256 = cpu_sub_vec256,
    };
    register_op("Sub",
                OP_SUB,
                OP_ELEMENT_WISE,
                OP_ARITY_BINARY,
                validate_binary_element_wise,
                select_cpu_kernel(&sub_kernels),
                select_gpu_kernel(cuda_sub));

    // === Mul Kernels ===
    CpuKernels mul_kernels = {
            .scalar = cpu_mul_scalar,
            .vec128 = cpu_mul_vec128,
            .vec256 = cpu_mul_vec256,
    };
    register_op("Mul",
                OP_MUL,
                OP_ELEMENT_WISE,
                OP_ARITY_BINARY,
                validate_binary_element_wise,
                select_cpu_kernel(&mul_kernels),
                select_gpu_kernel(cuda_mul));

    // === Div Kernels ===
    CpuKernels div_kernels = {
            .scalar = cpu_div_scalar,
            .vec128 = cpu_div_vec128,
            .vec256 = cpu_div_vec256,
    };
    register_op("Div",
                OP_DIV,
                OP_ELEMENT_WISE,
                OP_ARITY_BINARY,
                validate_binary_element_wise,
                select_cpu_kernel(&div_kernels),
                select_gpu_kernel(cuda_div));

    // === Pow Kernels ===
    CpuKernels pow_kernels = {
            .scalar = cpu_pow_scalar,
            .vec128 = cpu_pow_vec128,
            .vec256 = cpu_pow_vec256,
    };
    register_op("Pow",
                OP_POW,
                OP_ELEMENT_WISE,
                OP_ARITY_BINARY,
                validate_binary_element_wise,
                select_cpu_kernel(&pow_kernels),
                select_gpu_kernel(cuda_pow));

    // === Matmul Kernels ===
    CpuKernels matmul_kernels = {
            .scalar = cpu_gemm_scalar,
            .vec128 = cpu_gemm_vec128,
            .vec256 = cpu_gemm_vec256,
    };
    register_op("Matmul",
                OP_MATMUL,
                OP_GEMM,
                OP_ARITY_BINARY,
                validate_matmul,
                select_cpu_kernel(&matmul_kernels),
                select_gpu_kernel(cuda_gemm));
}
