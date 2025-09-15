#include "element_wise.h"
#include "functors.hpp"
#include "impl.hpp"

// Add operations
int cpu_add_scalar(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_scalar_impl<AddOp>(inputs, out, numel);
}

int cpu_add_vec128(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec128_impl<AddOp>(inputs, out, numel);
}

int cpu_add_vec256(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec256_impl<AddOp>(inputs, out, numel);
}

// Sub operations
int cpu_sub_scalar(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_scalar_impl<SubOp>(inputs, out, numel);
}

int cpu_sub_vec128(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec128_impl<SubOp>(inputs, out, numel);
}

int cpu_sub_vec256(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec256_impl<SubOp>(inputs, out, numel);
}

// Mul operations
int cpu_mul_scalar(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_scalar_impl<MulOp>(inputs, out, numel);
}

int cpu_mul_vec128(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec128_impl<MulOp>(inputs, out, numel);
}

int cpu_mul_vec256(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec256_impl<MulOp>(inputs, out, numel);
}

// Div operations
int cpu_div_scalar(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_scalar_impl<DivOp>(inputs, out, numel);
}

int cpu_div_vec128(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec128_impl<DivOp>(inputs, out, numel);
}

int cpu_div_vec256(const double** inputs, double* out, size_t numel) {
    return cpu_element_wise_vec256_impl<DivOp>(inputs, out, numel);
}