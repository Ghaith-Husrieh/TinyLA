#include "op_wrapper.h"
#include "dispatcher.h"
#include "tinyla/tensor.h"
#include <stdio.h>

static int validate_shapes(const Tensor** inputs, size_t n_inputs, const Tensor* out) {
    if (n_inputs == 0) {
        fprintf(stderr, "No input tensors provided.\n");
        return -1;
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
    if (!a || !b || !out) {
        fprintf(stderr, "add_op_wrapper: NULL tensor\n");
        return -1;
    }

    if (a->device != b->device || a->device != out->device) {
        fprintf(stderr, "add_op_wrapper: tensors must be on same device\n");
        return -1;
    }

    if (out->numel == 0) {
        return 0;
    }

    const Tensor* inputs[2] = {a, b};
    if (validate_shapes(inputs, 2, out) != 0) {
        return -1;
    }

    return dispatch_op(OP_ADD, out, inputs, 2);
}
