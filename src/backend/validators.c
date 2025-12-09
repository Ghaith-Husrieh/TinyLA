#include "validators.h"
#include <stdio.h>

static inline int base_validation(const tensor_desc** inputs, size_t n_inputs) {
    if (n_inputs == 0) {
        fprintf(stderr, "No input tensors provided.\n");
        return -1;
    }

    for (size_t i = 0; i < n_inputs; i++) {
        if (!inputs[i]) {
            fprintf(stderr, "Input tensor cannot be NULL.\n");
            return -1;
        }
    }

    device device = inputs[0]->device;
    for (size_t i = 1; i < n_inputs; i++) {
        if (inputs[i]->device != device) {
            fprintf(stderr, "Input tensors must be on the same device.\n");
            return -1;
        }
    }

    size_t ndim = inputs[0]->ndim;
    for (size_t i = 1; i < n_inputs; i++) {
        if (inputs[i]->ndim != ndim) {
            fprintf(stderr, "Input tensors must have the same number of dimensions.\n");
            return -1;
        }
    }

    return 0;
}

static inline int validate_element_wise(const tensor_desc** inputs, size_t n_inputs) {
    if (base_validation(inputs, n_inputs) != 0) {
        return -1;
    }

    for (size_t dim = 0; dim < inputs[0]->ndim; dim++) {
        size_t dim_numel = inputs[0]->shape[dim];
        for (size_t i = 1; i < n_inputs; i++) {
            if (inputs[i]->shape[dim] != dim_numel) {
                fprintf(stderr,
                        "Input tensor %zu has numel %zu at dimension %zu, expected %zu.\n",
                        i,
                        inputs[i]->shape[dim],
                        dim,
                        dim_numel);
                return -1;
            }
        }
    }

    return 0;
}

int validate_unary_element_wise(const tensor_desc** inputs, size_t n_inputs) {
    if (n_inputs != 1) {
        fprintf(stderr, "Unary element-wise operation requires 1 input tensor, got %zu. instead.\n", n_inputs);
        return -1;
    }

    return validate_element_wise(inputs, n_inputs);
}

int validate_binary_element_wise(const tensor_desc** inputs, size_t n_inputs) {
    if (n_inputs != 2) {
        fprintf(stderr, "Binary element-wise operation requires 2 input tensors, got %zu. instead.\n", n_inputs);
        return -1;
    }

    return validate_element_wise(inputs, n_inputs);
}

int validate_matmul(const tensor_desc** inputs, size_t n_inputs) {
    if (base_validation(inputs, n_inputs) != 0) {
        return -1;
    }

    if (n_inputs != 2) {
        fprintf(stderr, "Matmul operation requires 2 input tensors, got %zu. instead.\n", n_inputs);
        return -1;
    }

    if (inputs[0]->ndim < 2 || inputs[1]->ndim < 2) {
        fprintf(stderr, "Input tensors must have at least 2 dimensions.\n");
        return -1;
    }

    for (size_t i = 0; i < inputs[0]->ndim - 2; i++) {
        if (inputs[0]->shape[i] != inputs[1]->shape[i]) {
            fprintf(stderr,
                    "Input batch dimension mismatch at dimension %zu: %zu != %zu.\n",
                    i,
                    inputs[0]->shape[i],
                    inputs[1]->shape[i]);
            return -1;
        }
    }

    if (inputs[0]->shape[inputs[0]->ndim - 1] != inputs[1]->shape[inputs[1]->ndim - 2]) {
        fprintf(stderr,
                "Input matrix dimension mismatch, (%zu, %zu) @ (%zu, %zu) cannot be multiplied.\n",
                inputs[0]->shape[inputs[0]->ndim - 2],
                inputs[0]->shape[inputs[0]->ndim - 1],
                inputs[1]->shape[inputs[1]->ndim - 2],
                inputs[1]->shape[inputs[1]->ndim - 1]);
        return -1;
    }

    return 0;
}
