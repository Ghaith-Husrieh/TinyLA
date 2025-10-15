#pragma once
#include "tla_alloc.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FROM_DATA,
    UNINITIALIZED,
    ZEROS,
    ONES,
} buffer_init_mode;

typedef struct {
    double* buffer;

    size_t* shape;
    size_t ndim;
    size_t numel;
    device device;
} tensor_desc;

tensor_desc* tensor_desc_create(const double* data, const size_t* shape, size_t ndim, device device,
                                buffer_init_mode init_mode);
tensor_desc* tensor_desc_to_device(tensor_desc* desc, device device, bool force_copy);
void tensor_desc_free(tensor_desc** desc_ptr);
void tensor_desc_print(const tensor_desc* desc);

#ifdef __cplusplus
}
#endif