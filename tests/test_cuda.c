#include "tinyla/tinyla.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

static void test_tensor_creation_gpu() {
    printf("Testing GPU tensor creation...\n");

    size_t shape[2] = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};

    Tensor* t = tensor(data, shape, 2, DEVICE_GPU);
    assert(t && t->ndim == 2 && t->numel == 6 && t->device == DEVICE_GPU);

    Tensor* host_out = to_cpu(t);
    for (size_t i = 0; i < 6; i++)
        assert(host_out->buffer[i] == data[i]);

    t->free(&t);
    host_out->free(&host_out);
    printf("✓ GPU tensor creation passed\n");
}

static void test_tensor_print_gpu() {
    printf("Testing GPU tensor print...\n");

    size_t shape[2] = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};
    Tensor* t = tensor(data, shape, 2, DEVICE_GPU);

    printf("Expected output: 2x3 tensor with values 1..6\n");
    t->print(t);

    t->free(&t);
    printf("✓ GPU tensor print passed\n");
}

static void test_add_gpu() {
    printf("Testing GPU add operation...\n");

    size_t shape[1] = {4};
    double a_data[4] = {1, 2, 3, 4};
    double b_data[4] = {5, 6, 7, 8};

    Tensor* a = tensor(a_data, shape, 1, DEVICE_GPU);
    Tensor* b = tensor(b_data, shape, 1, DEVICE_GPU);
    Tensor* out = empty_tensor(shape, 1, DEVICE_GPU);

    int ret = add(out, a, b);
    assert(ret == 0);

    Tensor* host_out = to_cpu(out);
    for (size_t i = 0; i < 4; i++)
        assert(host_out->buffer[i] == a_data[i] + b_data[i]);

    a->free(&a);
    b->free(&b);
    out->free(&out);
    host_out->free(&host_out);
    printf("✓ GPU add operation passed\n");
}

int main() {
    tinyla_init();

    printf("Running GPU tests\n================\n");
    test_tensor_creation_gpu();
    test_tensor_print_gpu();
    test_add_gpu();
    printf("\nAll GPU tests passed! 🎉\n");
    return 0;
}
