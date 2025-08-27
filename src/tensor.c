#include "tinyla/tensor.h"
#include "memory/tla_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/init/ones.h"
#include <cuda_runtime.h>
#endif

#define EDGEITEMS 3

typedef enum {
    TENSOR_WITH_DATA,
    TENSOR_UNINITIALIZED,
    TENSOR_ZEROS,
    TENSOR_ONES,
} TensorInitMode;

static void _tensor_print_recursive(const Tensor* tensor, size_t dim, size_t offset, int indent) {
    size_t size = tensor->shape[dim];

    // ===== Base case: last dimension =====
    if (dim == tensor->ndim - 1) {
        printf("[");
        if (size > 2 * EDGEITEMS) {
            for (size_t i = 0; i < EDGEITEMS; i++)
                printf("%.4g, ", tensor->buffer[offset + i]);

            printf("..., ");

            for (size_t i = size - EDGEITEMS; i < size; i++) {
                printf("%.4g", tensor->buffer[offset + i]);
                if (i != size - 1)
                    printf(", ");
            }
        } else {
            for (size_t i = 0; i < size; i++) {
                printf("%.4g", tensor->buffer[offset + i]);
                if (i != size - 1)
                    printf(", ");
            }
        }
        printf("]");
        return;
    }

    // ===== Recursive case: ND slice =====
    printf("[");
    size_t stride = 1;
    for (size_t i = dim + 1; i < tensor->ndim; i++)
        stride *= tensor->shape[i];

    if (size > 2 * EDGEITEMS) {
        for (size_t i = 0; i < EDGEITEMS; i++) {
            if (i > 0) {
                printf("\n");
                for (int j = 0; j < indent + 1; j++)
                    printf(" ");
            }
            _tensor_print_recursive(tensor, dim + 1, offset + i * stride, indent + 1);
            printf(",");
        }

        printf("\n");
        for (int j = 0; j < indent + 1; j++)
            printf(" ");
        printf("...\n");

        for (size_t i = size - EDGEITEMS; i < size; i++) {
            for (int j = 0; j < indent + 1; j++)
                printf(" ");
            _tensor_print_recursive(tensor, dim + 1, offset + i * stride, indent + 1);
            if (i != size - 1)
                printf(",\n");
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            if (i > 0) {
                printf("\n");
                for (int j = 0; j < indent + 1; j++)
                    printf(" ");
            }
            _tensor_print_recursive(tensor, dim + 1, offset + i * stride, indent + 1);
            if (i != size - 1)
                printf(",");
        }
    }
    printf("]");
}

void tensor_print(const Tensor* tensor) {
    // ===== Special case: NULL tensor =====
    if (!tensor) {
        printf("tensor(NULL)\n");
        return;
    }

    // ===== Special case: 0D scalar =====
    if (tensor->ndim == 0) {
        double scalar;
        if (tensor->device == DEVICE_GPU) {
            if (tla_memcpy_safe(&scalar, tensor->buffer, sizeof(double), TLA_MEMCPY_DEVICE_TO_HOST) != 0) {
                fprintf(stderr, "Failed to copy tensor buffer to host\n");
                return;
            }
        } else
            scalar = tensor->buffer[0];

        printf("tensor(%.4g, device=%s, dtype=float64)\n", scalar, tensor->device == DEVICE_CPU ? "cpu" : "cuda");
        return;
    }

    // ===== Special case: empty tensor =====
    if (tensor->numel == 0) {
        printf("tensor([], shape=(");
        for (size_t i = 0; i < tensor->ndim; i++) {
            printf("%zu", tensor->shape[i]);
            if (i + 1 < tensor->ndim)
                printf(", ");
        }
        printf("), device=%s, dtype=float64)\n", tensor->device == DEVICE_CPU ? "cpu" : "cuda");
        return;
    }

    // ===== Handle CUDA tensors =====
    double* host_buffer = NULL;
    if (tensor->device == DEVICE_GPU) {
        host_buffer = malloc(tensor->numel * sizeof(double)); // TODO: use pinned memory instead of malloc
        if (tla_memcpy_safe(host_buffer, tensor->buffer, tensor->numel * sizeof(double), TLA_MEMCPY_DEVICE_TO_HOST) !=
            0) {
            fprintf(stderr, "Failed to copy tensor buffer to host\n");
            return;
        }
    }

    Tensor host_tensor = *tensor;
    if (host_buffer) {
        host_tensor.buffer = host_buffer;
    }
    host_tensor.device = DEVICE_CPU;

    // ===== General case =====
    printf("tensor(");
    _tensor_print_recursive(&host_tensor, 0, 0, 7);
    printf(", device=%s, dtype=float64)\n", tensor->device == DEVICE_CPU ? "cpu" : "cuda");

    if (host_buffer) {
        free(host_buffer);
    }
}

void tensor_free(Tensor** tensor_ptr) {
    if (!tensor_ptr || !*tensor_ptr)
        return;

    Tensor* tensor = *tensor_ptr;

    if (tensor->buffer) {
        tla_free(tensor->device, tensor->buffer);
        tensor->buffer = NULL;
    }

    if (tensor->shape) {
        free(tensor->shape);
        tensor->shape = NULL;
    }

    free(tensor);
    *tensor_ptr = NULL;
}

static Tensor* _tensor_init(const double* data, const size_t* shape, size_t ndim, Device device,
                            TensorInitMode init_mode) {
    Tensor* tensor = calloc(1, sizeof(Tensor));
    if (!tensor) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->device = device;

    size_t numel = 1;

    if (ndim == 0) {
        tensor->numel = 1;
        tensor->shape = NULL;
        tensor->buffer = tla_malloc(device, sizeof(double));
        if (!tensor->buffer) {
            fprintf(stderr, "Failed to allocate memory for buffer\n");
            goto cleanup;
        }
    } else {
        if (!shape) {
            fprintf(stderr, "Shape must be provided\n");
            goto cleanup;
        }

        for (size_t i = 0; i < ndim; i++) {
            numel *= shape[i];
        }
        tensor->numel = numel;

        tensor->shape = malloc(ndim * sizeof(size_t));
        if (!tensor->shape) {
            fprintf(stderr, "Failed to allocate memory for shape\n");
            goto cleanup;
        }
        memcpy(tensor->shape, shape, ndim * sizeof(size_t));

        if (numel == 0) {
            tensor->buffer = tla_malloc(device, 0);
            if (!tensor->buffer) {
                fprintf(stderr, "Failed to allocate memory for buffer\n");
                goto cleanup;
            }
            goto finalize;
        }

        tensor->buffer = tla_malloc(device, numel * sizeof(double));
        if (!tensor->buffer) {
            fprintf(stderr, "Failed to allocate memory for buffer\n");
            goto cleanup;
        }
    }

    switch (init_mode) {
    case TENSOR_WITH_DATA:
        if (!data) {
            fprintf(stderr, "Data must be provided for TENSOR_WITH_DATA\n");
            goto cleanup;
        }
        if (device == DEVICE_GPU) {
            if (tla_memcpy_safe(tensor->buffer, data, numel * sizeof(double), TLA_MEMCPY_HOST_TO_DEVICE) != 0) {
                fprintf(stderr, "Failed to copy data to tensor buffer on device %d\n", device);
                goto cleanup;
            }
        } else {
            if (tla_memcpy_safe(tensor->buffer, data, numel * sizeof(double), TLA_MEMCPY_HOST_TO_HOST) != 0) {
                fprintf(stderr, "Failed to copy data to tensor buffer on device %d\n", device);
                goto cleanup;
            }
        }
        break;

    case TENSOR_UNINITIALIZED:
        break;

    case TENSOR_ZEROS:
        if (tla_memset_safe(device, tensor->buffer, 0, numel * sizeof(double)) != 0) {
            fprintf(stderr, "Failed to zero tensor buffer on device %d\n", device);
            goto cleanup;
        }
        break;

    case TENSOR_ONES: {
        if (device == DEVICE_CPU) {
            for (size_t i = 0; i < numel; i++)
                tensor->buffer[i] = 1.0;
        }
#ifdef TINYLA_CUDA_ENABLED
        else if (device == DEVICE_GPU) {
            launch_ones_kernel(tensor->buffer, numel);
        }
#endif
    } break;

    default:
        fprintf(stderr, "Invalid initialization mode\n");
        goto cleanup;
    }

finalize:
    tensor->free = tensor_free;
    tensor->print = tensor_print;
    return tensor;

cleanup:
    tensor_free(&tensor);
    return NULL;
}

Tensor* tensor(const double* data, const size_t* shape, size_t ndim, Device device) {
    return _tensor_init(data, shape, ndim, device, TENSOR_WITH_DATA);
}
Tensor* empty_tensor(const size_t* shape, size_t ndim, Device device) {
    return _tensor_init(NULL, shape, ndim, device, TENSOR_UNINITIALIZED);
}
Tensor* zeroes_tensor(const size_t* shape, size_t ndim, Device device) {
    return _tensor_init(NULL, shape, ndim, device, TENSOR_ZEROS);
}
Tensor* ones_tensor(const size_t* shape, size_t ndim, Device device) {
    return _tensor_init(NULL, shape, ndim, device, TENSOR_ONES);
}
