#pragma once
#include "common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cpu_gemm_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cpu_gemm_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);
int cpu_gemm_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif
