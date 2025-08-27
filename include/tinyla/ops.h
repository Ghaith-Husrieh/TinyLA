#pragma once
#include "../src/backend/op_wrapper.h"
#include "tinyla/tinyla.h"

/**
 * @brief Performs element-wise addition of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int add(Tensor* out, const Tensor* a, const Tensor* b) { return add_op_wrapper(out, a, b); }
