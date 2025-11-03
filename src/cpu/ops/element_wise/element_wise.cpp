#include "element_wise.h"
#include "functors.hpp"
#include "impl.hpp"

// Add operations
int cpu_add_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_scalar_impl<AddOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_add_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec128_impl<AddOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_add_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec256_impl<AddOp>(out, inputs[0], inputs[1], out->numel);
}

// Sub operations
int cpu_sub_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_scalar_impl<SubOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_sub_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec128_impl<SubOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_sub_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec256_impl<SubOp>(out, inputs[0], inputs[1], out->numel);
}

// Mul operations
int cpu_mul_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_scalar_impl<MulOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_mul_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec128_impl<MulOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_mul_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec256_impl<MulOp>(out, inputs[0], inputs[1], out->numel);
}

// Div operations
int cpu_div_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_scalar_impl<DivOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_div_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec128_impl<DivOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_div_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec256_impl<DivOp>(out, inputs[0], inputs[1], out->numel);
}

// Pow operations
int cpu_pow_scalar(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_scalar_impl<PowOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_pow_vec128(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec128_impl<PowOp>(out, inputs[0], inputs[1], out->numel);
}
int cpu_pow_vec256(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cpu_element_wise_vec256_impl<PowOp>(out, inputs[0], inputs[1], out->numel);
}
