#include "tinyla/tinyla.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

static void test_tensor_creation_gpu() {
    printf("Testing GPU tensor creation...\n");

    std::vector<size_t> shape = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};

    tla::Tensor t = tla::Tensor::tensor(data, shape, tla::Device::CUDA);
    assert(t.ndim() == 2 && t.numel() == 6 && t.device() == tla::Device::CUDA);

    tla::Tensor host_out = t.to_cpu();
    for (size_t i = 0; i < 6; i++)
        assert(host_out.data()[i] == data[i]);

    printf("✓ GPU tensor creation passed\n");
}

static void test_tensor_print_gpu() {
    printf("Testing GPU tensor print...\n");

    std::vector<size_t> shape = {2, 3};
    double data[6] = {1, 2, 3, 4, 5, 6};
    tla::Tensor t = tla::Tensor::tensor(data, shape, tla::Device::CUDA);

    printf("Expected output: 2x3 tensor with values 1..6\n");
    t.print();

    printf("✓ GPU tensor print passed\n");
}

static void test_add_gpu() {
    printf("Testing GPU add operation...\n");

    std::vector<size_t> shape = {4};
    double a_data[4] = {1, 2, 3, 4};
    double b_data[4] = {5, 6, 7, 8};

    tla::Tensor a = tla::Tensor::tensor(a_data, shape, tla::Device::CUDA);
    tla::Tensor b = tla::Tensor::tensor(b_data, shape, tla::Device::CUDA);
    tla::Tensor out = a.add(b);

    tla::Tensor host_out = out.to_cpu();
    for (size_t i = 0; i < 4; i++)
        assert(host_out.data()[i] == a_data[i] + b_data[i]);

    printf("✓ GPU add operation passed\n");
}

int main() {
    tinyla_init();

    printf("Running GPU tests\n================\n");
    test_tensor_creation_gpu();
    test_tensor_print_gpu();
    test_add_gpu();
    printf("\nAll GPU tests passed!\n");
    return 0;
}
