#pragma once
#include "tinyla/tensor.h"

static int validate_tensors(const Tensor** inputs, size_t n_inputs, const Tensor* out, const char* op_name);

// Operations
int add_op_wrapper(Tensor* out, const Tensor* a, const Tensor* b);
int sub_op_wrapper(Tensor* out, const Tensor* a, const Tensor* b);