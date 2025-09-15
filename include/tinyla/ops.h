#pragma once
#include "../src/backend/op_wrapper.h"
#include "tinyla/tinyla.h"

#ifdef __cplusplus
extern "C" {
#endif
// === C interface ===

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
static inline int tla_add(Tensor* out, const Tensor* a, const Tensor* b) { return add_op_wrapper(out, a, b); }

/**
 * @brief Performs element-wise subtraction of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int tla_sub(Tensor* out, const Tensor* a, const Tensor* b) { return sub_op_wrapper(out, a, b); }

/**
 * @brief Performs element-wise multiplication of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int tla_mul(Tensor* out, const Tensor* a, const Tensor* b) { return mul_op_wrapper(out, a, b); }

/**
 * @brief Performs element-wise division of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int tla_div(Tensor* out, const Tensor* a, const Tensor* b) { return div_op_wrapper(out, a, b); }

#ifdef __cplusplus
}
// === C++ interface ===

namespace tla {
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

/**
 * @brief Performs element-wise subtraction of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int sub(Tensor* out, const Tensor* a, const Tensor* b) { return sub_op_wrapper(out, a, b); }

/**
 * @brief Performs element-wise multiplication of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int mul(Tensor* out, const Tensor* a, const Tensor* b) { return mul_op_wrapper(out, a, b); }

/**
 * @brief Performs element-wise division of two tensors.
 *
 * The tensors must have the same shape and reside on the same device.
 *
 * @param out Pointer to the output tensor.
 * @param a   Pointer to the first input tensor.
 * @param b   Pointer to the second input tensor.
 *
 * @return 0 on success, -1 on error (e.g., shape/device mismatch).
 */
static inline int div(Tensor* out, const Tensor* a, const Tensor* b) { return div_op_wrapper(out, a, b); }
} // namespace tla

#endif
