#include "tensor_desc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/init/ones.h"
#include <cuda_runtime.h>
#endif

#define EDGEITEMS 3

void tensor_desc_free(tensor_desc** desc_ptr) {
    if (!desc_ptr || !*desc_ptr)
        return;

    tensor_desc* desc = *desc_ptr;

    if (desc->buffer) {
        tla_free(desc->device, desc->buffer);
        desc->buffer = NULL;
    }

    if (desc->shape) {
        free(desc->shape);
        desc->shape = NULL;
    }

    free(desc);
    *desc_ptr = NULL;
}

tensor_desc* tensor_desc_create(const double* data, const size_t* shape, size_t ndim, device device,
                                buffer_init_mode init_mode) {
    tensor_desc* desc = calloc(1, sizeof(tensor_desc));
    if (!desc) {
        fprintf(stderr, "Failed to allocate memory for tensor_desc\n");
        return NULL;
    }

    desc->device = device;
    desc->ndim = ndim;

    size_t numel = 1;

    if (ndim == 0) {
        desc->numel = 1;
        desc->shape = NULL;
        desc->buffer = tla_malloc(device, sizeof(double));
        if (!desc->buffer) {
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
        desc->numel = numel;

        desc->shape = malloc(ndim * sizeof(size_t));
        if (!desc->shape) {
            fprintf(stderr, "Failed to allocate memory for shape\n");
            goto cleanup;
        }
        memcpy(desc->shape, shape, ndim * sizeof(size_t));

        if (numel == 0) {
            desc->buffer = tla_malloc(device, 0);
            if (!desc->buffer) {
                fprintf(stderr, "Failed to allocate memory for buffer\n");
                goto cleanup;
            }
            goto finalize;
        }

        desc->buffer = tla_malloc(device, numel * sizeof(double));
        if (!desc->buffer) {
            fprintf(stderr, "Failed to allocate memory for buffer\n");
            goto cleanup;
        }
    }

    switch (init_mode) {
    case FROM_DATA:
        if (!data) {
            fprintf(stderr, "Data must be provided for init mode FROM_DATA\n");
            goto cleanup;
        }
        if (device == DEVICE_CUDA) {
            if (tla_memcpy_safe(desc->buffer, data, numel * sizeof(double), TLA_MEMCPY_HOST_TO_DEVICE) != 0) {
                fprintf(stderr, "Failed to copy data to tensor buffer on device %d\n", device);
                goto cleanup;
            }
        } else {
            if (tla_memcpy_safe(desc->buffer, data, numel * sizeof(double), TLA_MEMCPY_HOST_TO_HOST) != 0) {
                fprintf(stderr, "Failed to copy data to tensor buffer on device %d\n", device);
                goto cleanup;
            }
        }
        break;
    case UNINITIALIZED:
        break;
    case ZEROS:
        if (tla_memset_safe(device, desc->buffer, 0, numel * sizeof(double)) != 0) {
            fprintf(stderr, "Failed to zero tensor buffer on device %d\n", device);
            goto cleanup;
        }
        break;
    case ONES:
        if (device == DEVICE_CPU) {
            for (size_t i = 0; i < numel; i++)
                desc->buffer[i] = 1.0;
        }
#ifdef TINYLA_CUDA_ENABLED
        else if (device == DEVICE_CUDA) {
            launch_ones_kernel(desc->buffer, numel);
        }
#endif
        break;
    default:
        fprintf(stderr, "Invalid initialization mode\n");
        goto cleanup;
    }

finalize:
    return desc;

cleanup:
    tensor_desc_free(&desc);
    return NULL;
}

tensor_desc* tensor_desc_to_device(tensor_desc* desc, device device, bool force_copy) {
    if (!desc) {
        return NULL;
    }
    if (desc->device == device && !force_copy) {
        return (tensor_desc*)desc;
    }

    tensor_desc* out = tensor_desc_create(NULL, desc->shape, desc->ndim, device, UNINITIALIZED);

    TLAMemcpyKind kind;
    if (desc->device == DEVICE_CPU && device == DEVICE_CPU) {
        kind = TLA_MEMCPY_HOST_TO_HOST;
#ifdef TINYLA_CUDA_ENABLED
    } else if (desc->device == DEVICE_CUDA && device == DEVICE_CUDA) {
        kind = TLA_MEMCPY_DEVICE_TO_DEVICE;
    } else if (desc->device == DEVICE_CPU && device == DEVICE_CUDA) {
        kind = TLA_MEMCPY_HOST_TO_DEVICE;
    } else if (desc->device == DEVICE_CUDA && device == DEVICE_CPU) {
        kind = TLA_MEMCPY_DEVICE_TO_HOST;
    }
#endif

    if (tla_memcpy_safe(out->buffer, desc->buffer, desc->numel * sizeof(double), kind) != 0) {
        fprintf(stderr, "Failed to copy tensor buffer in to_device()\n");
        tensor_desc_free(&out);
        return NULL;
    }

    return out;
}

static void tensor_desc_print_recursive(const tensor_desc* desc, size_t dim, size_t offset, int indent) {
    size_t size = desc->shape[dim];

    // ===== Base case: last dimension =====
    if (dim == desc->ndim - 1) {
        printf("[");
        if (size > 2 * EDGEITEMS) {
            for (size_t i = 0; i < EDGEITEMS; i++)
                printf("%.4g, ", desc->buffer[offset + i]);

            printf("..., ");

            for (size_t i = size - EDGEITEMS; i < size; i++) {
                printf("%.4g", desc->buffer[offset + i]);
                if (i != size - 1)
                    printf(", ");
            }
        } else {
            for (size_t i = 0; i < size; i++) {
                printf("%.4g", desc->buffer[offset + i]);
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
    for (size_t i = dim + 1; i < desc->ndim; i++)
        stride *= desc->shape[i];

    if (size > 2 * EDGEITEMS) {
        for (size_t i = 0; i < EDGEITEMS; i++) {
            if (i > 0) {
                printf("\n");
                for (int j = 0; j < indent + 1; j++)
                    printf(" ");
            }
            tensor_desc_print_recursive(desc, dim + 1, offset + i * stride, indent + 1);
            printf(",");
        }

        printf("\n");
        for (int j = 0; j < indent + 1; j++)
            printf(" ");
        printf("...\n");

        for (size_t i = size - EDGEITEMS; i < size; i++) {
            for (int j = 0; j < indent + 1; j++)
                printf(" ");
            tensor_desc_print_recursive(desc, dim + 1, offset + i * stride, indent + 1);
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
            tensor_desc_print_recursive(desc, dim + 1, offset + i * stride, indent + 1);
            if (i != size - 1)
                printf(",");
        }
    }
    printf("]");
}

void tensor_desc_print(const tensor_desc* desc) {
    // ===== Special case: NULL tensor =====
    if (!desc) {
        printf("tensor(NULL)\n");
        return;
    }

    // ===== Special case: empty tensor =====
    if (desc->numel == 0) {
        printf("tensor([], shape=(");
        for (size_t i = 0; i < desc->ndim; i++) {
            printf("%zu", desc->shape[i]);
            if (i + 1 < desc->ndim)
                printf(", ");
        }
        printf("), device=%s, dtype=float64)\n", desc->device == DEVICE_CPU ? "cpu" : "cuda");
        return;
    }

    // ===== Handle CUDA tensors =====
    tensor_desc* host_desc = tensor_desc_to_device(desc, DEVICE_CPU, false);
    if (!host_desc) {
        fprintf(stderr, "Failed to copy tensor to host for printing\n");
        return;
    }

    // ===== Special case: 0D scalar =====
    if (desc->ndim == 0) {
        double scalar = host_desc->buffer[0];
        if (host_desc != desc) {
            tensor_desc_free(&host_desc);
        }

        printf("tensor(%.4g, device=%s, dtype=float64)\n", scalar, desc->device == DEVICE_CPU ? "cpu" : "cuda");
        return;
    }

    // ===== General case =====
    printf("tensor(");
    tensor_desc_print_recursive(host_desc, 0, 0, 7);
    printf(", device=%s, dtype=float64)\n", desc->device == DEVICE_CPU ? "cpu" : "cuda");

    if (host_desc != desc) {
        tensor_desc_free(&host_desc);
    }
}
