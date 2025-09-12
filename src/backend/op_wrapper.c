#include "op_wrapper.h"
#include "dispatcher.h"
#include "tinyla/tensor.h"
#include <stdio.h>

static int validate_tensors(const Tensor** inputs, size_t n_inputs, const Tensor* out, const char* op_name) {
    if (n_inputs == 0) {
        fprintf(stderr, "No input tensors provided.\n");
        return -1;
    }

    for (size_t i = 0; i < n_inputs; i++) {
        if (!inputs[i]) {
            fprintf(stderr, "%s: NULL tensor\n", op_name);
            return -1;
        }
    }

    if (!out) {
        fprintf(stderr, "%s: NULL output tensor\n", op_name);
        return -1;
    }

    Device device = inputs[0]->device;
    for (size_t i = 1; i < n_inputs; i++) {
        if (inputs[i]->device != device) {
            fprintf(stderr, "%s: input tensor %zu device mismatch: got %d, expected %d\n", op_name, i,
                    inputs[i]->device, device);
            return -1;
        }
    }

    if (out->device != device) {
        fprintf(stderr, "%s: output tensor device mismatch: got %d, expected %d\n", op_name, out->device, device);
        return -1;
    }

    if (out->numel == 0) {
        return 0;
    }

    size_t ndim = inputs[0]->ndim;
    for (size_t i = 1; i < n_inputs; i++) {
        if (inputs[i]->ndim != ndim) {
            fprintf(stderr, "Shape mismatch: input %zu has ndim %zu, expected %zu\n", i, inputs[i]->ndim, ndim);
            return -1;
        }
    }

    if (out->ndim != ndim) {
        fprintf(stderr, "Output tensor ndim %zu does not match input ndim %zu\n", out->ndim, ndim);
        return -1;
    }

    for (size_t dim = 0; dim < ndim; dim++) {
        size_t dim_numel = inputs[0]->shape[dim];
        for (size_t i = 1; i < n_inputs; i++) {
            if (inputs[i]->shape[dim] != dim_numel) {
                fprintf(stderr, "Shape mismatch at dim %zu: input %zu has size %zu, expected %zu\n", dim, i,
                        inputs[i]->shape[dim], dim_numel);
                return -1;
            }
        }

        if (out->shape[dim] != dim_numel) {
            fprintf(stderr, "Output tensor shape mismatch at dim %zu: got %zu, expected %zu\n", dim, out->shape[dim],
                    dim_numel);
            return -1;
        }
    }

    return 0;
}

int add_op_wrapper(Tensor* out, const Tensor* a, const Tensor* b) {
    const Tensor* inputs[2] = {a, b};
    if (validate_tensors(inputs, 2, out, "add_op_wrapper") != 0) {
        return -1;
    }

    return dispatch_op(OP_ADD, out, inputs, 2);
}

int sub_op_wrapper(Tensor* out, const Tensor* a, const Tensor* b) {
    const Tensor* inputs[2] = {a, b};
    if (validate_tensors(inputs, 2, out, "sub_op_wrapper") != 0) {
        return -1;
    }

    return dispatch_op(OP_SUB, out, inputs, 2);
}