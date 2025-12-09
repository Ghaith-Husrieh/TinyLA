#pragma once
#include "../memory/tensor_desc.h"
#ifdef __cplusplus
extern "C" {
#endif

int validate_unary_element_wise(const tensor_desc** inputs, size_t n_inputs);
int validate_binary_element_wise(const tensor_desc** inputs, size_t n_inputs);
int validate_matmul(const tensor_desc** inputs, size_t n_inputs);

#ifdef __cplusplus
}
#endif
