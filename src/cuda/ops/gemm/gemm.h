#pragma once
#include "../../../cpu/ops/gemm/common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_gemm(Tensor* out, const Tensor** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif