#include "tinyla/tinyla.h"
#include <assert.h>
#include <stdio.h>

static void test_tensor_creation_cpu() {
    printf("Testing CPU tensor creation...\n");

    size_t shape[2] = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};

    Tensor* t = tensor(data, shape, 2, DEVICE_CPU);
    assert(t && t->ndim == 2 && t->numel == 6 && t->device == DEVICE_CPU);

    for (size_t i = 0; i < 6; i++)
        assert(t->buffer[i] == data[i]);

    t->free(&t);
    printf("✓ CPU tensor creation passed\n");
}

static void test_tensor_print_cpu() {
    printf("Testing CPU tensor print...\n");

    size_t shape[2] = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};
    Tensor* t = tensor(data, shape, 2, DEVICE_CPU);

    printf("Expected output: 2x3 tensor with values 1..6\n");
    t->print(t);

    t->free(&t);
    printf("✓ CPU tensor print passed\n");
}

static void test_add_cpu() {
    printf("Testing CPU add operation...\n");

    size_t shape[1] = {4};
    double a_data[4] = {1, 2, 3, 4};
    double b_data[4] = {5, 6, 7, 8};

    Tensor* a = tensor(a_data, shape, 1, DEVICE_CPU);
    Tensor* b = tensor(b_data, shape, 1, DEVICE_CPU);
    Tensor* out = empty_tensor(shape, 1, DEVICE_CPU);

    int ret = add(out, a, b);
    assert(ret == 0);

    for (size_t i = 0; i < 4; i++)
        assert(out->buffer[i] == a_data[i] + b_data[i]);

    a->free(&a);
    b->free(&b);
    out->free(&out);
    printf("✓ CPU add operation passed\n");
}

int main() {
    tinyla_init();

    printf("Running CPU tests\n================\n");
    test_tensor_creation_cpu();
    test_tensor_print_cpu();
    test_add_cpu();
    printf("\nAll CPU tests passed! 🎉\n");
    return 0;
}
