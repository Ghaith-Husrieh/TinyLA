#pragma once
#include "../../../cpu/ops/gemm/common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_gemm(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs);

#ifdef __cplusplus
}
#endif
