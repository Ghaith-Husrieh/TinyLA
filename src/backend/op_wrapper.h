#pragma once
#include "../memory/tensor_desc.h"
#include "dispatcher.h"

#ifdef __cplusplus
extern "C" {
#endif

static int validate_tensors(const tensor_desc** inputs, size_t n_inputs, const tensor_desc* out, const char* op_name);

// Macro for generating binary operation wrappers
#define BINARY_OP_WRAPPER(name, op_type)                                                                               \
    int name##_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b) {                              \
        const tensor_desc* inputs[2] = {a, b};                                                                         \
        if (validate_tensors(inputs, 2, out, #name "_op_wrapper") != 0) {                                              \
            return -1;                                                                                                 \
        }                                                                                                              \
        return dispatch_op(op_type, out, inputs, 2);                                                                   \
    }

// Operations
int add_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);
int sub_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);
int mul_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);
int div_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);
int pow_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);
int matmul_op_wrapper(tensor_desc* out, const tensor_desc* a, const tensor_desc* b);

#ifdef __cplusplus
}
#endif