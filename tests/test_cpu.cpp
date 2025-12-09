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

    tla::Tensor A = tla::Tensor::tensor(a_data, shape, tla::Device::CPU);
    tla::Tensor B = tla::Tensor::tensor(b_data, shape, tla::Device::CPU);
    tla::Tensor C = A.add(B);

    assert(C.ndim() == 1);
    assert(C.numel() == 4);
    assert(C.device() == tla::Device::CPU);

    const double* c_data = C.data();
    for (size_t i = 0; i < 4; i++)
        assert(c_data[i] == a_data[i] + b_data[i]);

    printf("✓ CPU add operation passed\n");
}

static void test_matmul_cpu_2d() {
    printf("Testing CPU matmul (2D)...\n");

    std::vector<size_t> a_shape = {2, 3};
    double a_data[6] = {1, 2, 3, 4, 5, 6};

    std::vector<size_t> b_shape = {3, 2};
    double b_data[6] = {7, 8, 9, 10, 11, 12};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CPU);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CPU);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 2);
    assert(C.numel() == 4);
    assert(C.device() == tla::Device::CPU);

    double expected[4] = {58, 64, 139, 154};

    const double* c_data = C.data();
    for (size_t i = 0; i < 4; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CPU matmul (2D) passed\n");
}

static void test_matmul_cpu_3d() {
    printf("Testing CPU matmul (3D batched)...\n");

    std::vector<size_t> a_shape = {2, 2, 2};
    double a_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<size_t> b_shape = {2, 2, 2};
    double b_data[8] = {9, 10, 11, 12, 13, 14, 15, 16};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CPU);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CPU);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 3);
    assert(C.numel() == 8);
    assert(C.device() == tla::Device::CPU);

    double expected[8] = {31, 34, 71, 78, 155, 166, 211, 226};

    const double* c_data = C.data();
    for (size_t i = 0; i < 8; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CPU matmul (3D batched) passed\n");
}

static void test_matmul_cpu_4d() {
    printf("Testing CPU matmul (4D batched)...\n");

    std::vector<size_t> a_shape = {2, 3, 2, 2};
    double a_data[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    std::vector<size_t> b_shape = {2, 3, 2, 2};
    double b_data[24] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                         113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CPU);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CPU);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 4);
    assert(C.numel() == 24);
    assert(C.device() == tla::Device::CPU);

    double expected[24] = {307,  310,  715,  722,  1167, 1178, 1591, 1606, 2091, 2110, 2531, 2554,
                           3079, 3106, 3535, 3566, 4131, 4166, 4603, 4642, 5247, 5290, 5735, 5782};

    const double* c_data = C.data();
    for (size_t i = 0; i < 24; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CPU matmul (4D batched) passed\n");
}

int main() {
    tinyla_init();

    printf("Running CPU tests\n================\n");
    test_tensor_creation_cpu();
    test_tensor_print_cpu();
    test_add_cpu();
    test_matmul_cpu_2d();
    test_matmul_cpu_3d();
    test_matmul_cpu_4d();
    printf("\nAll CPU tests passed!\n");
    return 0;
}
