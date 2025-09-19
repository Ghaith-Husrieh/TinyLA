#pragma once
#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cpu_gemm_scalar(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cpu_gemm_vec128(Tensor* out, const Tensor** inputs, const size_t n_inputs);
int cpu_gemm_vec256(Tensor* out, const Tensor** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif