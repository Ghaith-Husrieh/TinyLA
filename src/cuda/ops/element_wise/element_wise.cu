#include "element_wise.h"
#include "functors.cuh"
#include "impl.cuh"

// Add operations
int cuda_add(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cuda_element_wise_impl<AddOp>(out, inputs[0], inputs[1], out->numel);
}

// Sub operations
int cuda_sub(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cuda_element_wise_impl<SubOp>(out, inputs[0], inputs[1], out->numel);
}

// Mul operations
int cuda_mul(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cuda_element_wise_impl<MulOp>(out, inputs[0], inputs[1], out->numel);
}

// Div operations
int cuda_div(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cuda_element_wise_impl<DivOp>(out, inputs[0], inputs[1], out->numel);
}

// Pow operations
int cuda_pow(tensor_desc* out, const tensor_desc** inputs, const size_t n_inputs) {
    return cuda_element_wise_impl<PowOp>(out, inputs[0], inputs[1], out->numel);
}
