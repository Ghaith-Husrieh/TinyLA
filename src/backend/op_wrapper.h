#pragma once
#include "tinyla/tensor.h"

static int validate_shapes(const Tensor** inputs, size_t n_inputs, const Tensor* out);

// Operations
int add_op_wrapper(Tensor* out, const Tensor* a, const Tensor* b);