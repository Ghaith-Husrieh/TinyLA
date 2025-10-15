#include "tinyla/tinyla.h"
#include <assert.h>
#include <stdio.h>
#include <vector>

static void test_tensor_creation_cpu() {
    printf("Testing CPU tensor creation...\n");

    std::vector<size_t> shape = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};

    tla::Tensor t = tla::Tensor::tensor(data, shape, tla::Device::CPU);
    assert(t.ndim() == 2 && t.numel() == 6 && t.device() == tla::Device::CPU);

    for (size_t i = 0; i < 6; i++)
        assert(t.data()[i] == data[i]);

    printf("✓ CPU tensor creation passed\n");
}

static void test_tensor_print_cpu() {
    printf("Testing CPU tensor print...\n");

    std::vector<size_t> shape = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};
    tla::Tensor t = tla::Tensor::tensor(data, shape, tla::Device::CPU);

    printf("Expected output: 2x3 tensor with values 1..6\n");
    t.print();

    printf("✓ CPU tensor print passed\n");
}

static void test_add_cpu() {
    printf("Testing CPU add operation...\n");

    std::vector<size_t> shape = {4};
    double a_data[4] = {1, 2, 3, 4};
    double b_data[4] = {5, 6, 7, 8};

    tla::Tensor a = tla::Tensor::tensor(a_data, shape, tla::Device::CPU);
    tla::Tensor b = tla::Tensor::tensor(b_data, shape, tla::Device::CPU);
    tla::Tensor out = a.add(b);

    for (size_t i = 0; i < 4; i++)
        assert(out.data()[i] == a_data[i] + b_data[i]);

    printf("✓ CPU add operation passed\n");
}

int main() {
    tinyla_init();

    printf("Running CPU tests\n================\n");
    test_tensor_creation_cpu();
    test_tensor_print_cpu();
    test_add_cpu();
    printf("\nAll CPU tests passed!\n");
    return 0;
}
