#include "element_wise.h"
#include "functors.cuh"
#include "impl.cuh"

// Add operations
int cuda_add(const double** inputs, double* out, size_t numel) {
    return cuda_element_wise_impl<AddOp>(inputs[0], inputs[1], out, numel);
}

// Sub operations
int cuda_sub(const double** inputs, double* out, size_t numel) {
    return cuda_element_wise_impl<SubOp>(inputs[0], inputs[1], out, numel);
}

// Mul operations
int cuda_mul(const double** inputs, double* out, size_t numel) {
    return cuda_element_wise_impl<MulOp>(inputs[0], inputs[1], out, numel);
}

// Div operations
int cuda_div(const double** inputs, double* out, size_t numel) {
    return cuda_element_wise_impl<DivOp>(inputs[0], inputs[1], out, numel);
}

// Pow operations
int cuda_pow(const double** inputs, double* out, size_t numel) {
    return cuda_element_wise_impl<PowOp>(inputs[0], inputs[1], out, numel);
}