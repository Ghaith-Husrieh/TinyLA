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

    tla::Tensor A = tla::Tensor::tensor(a_data, shape, tla::Device::CUDA);
    tla::Tensor B = tla::Tensor::tensor(b_data, shape, tla::Device::CUDA);
    tla::Tensor C = A.add(B);

    assert(C.ndim() == 1);
    assert(C.numel() == 4);
    assert(C.device() == tla::Device::CUDA);

    tla::Tensor host_c = C.to_cpu();
    const double* c_data = host_c.data();
    for (size_t i = 0; i < 4; i++)
        assert(c_data[i] == a_data[i] + b_data[i]);

    printf("✓ GPU add operation passed\n");
}

static void test_matmul_cuda_2d() {
    printf("Testing CUDA matmul (2D)...\n");

    std::vector<size_t> a_shape = {2, 3};
    double a_data[6] = {1, 2, 3, 4, 5, 6};

    std::vector<size_t> b_shape = {3, 2};
    double b_data[6] = {7, 8, 9, 10, 11, 12};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CUDA);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CUDA);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 2);
    assert(C.numel() == 4);
    assert(C.device() == tla::Device::CUDA);

    double expected[4] = {58, 64, 139, 154};

    tla::Tensor host_c = C.to_cpu();
    const double* c_data = host_c.data();
    for (size_t i = 0; i < 4; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CUDA matmul (2D) passed\n");
}

static void test_matmul_cuda_3d() {
    printf("Testing CUDA matmul (3D batched)...\n");

    std::vector<size_t> a_shape = {2, 2, 2};
    double a_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<size_t> b_shape = {2, 2, 2};
    double b_data[8] = {9, 10, 11, 12, 13, 14, 15, 16};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CUDA);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CUDA);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 3);
    assert(C.numel() == 8);
    assert(C.device() == tla::Device::CUDA);

    double expected[8] = {31, 34, 71, 78, 155, 166, 211, 226};

    tla::Tensor host_c = C.to_cpu();
    const double* c_data = host_c.data();
    for (size_t i = 0; i < 8; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CUDA matmul (3D batched) passed\n");
}

static void test_matmul_cuda_4d() {
    printf("Testing CUDA matmul (4D batched)...\n");

    std::vector<size_t> a_shape = {2, 3, 2, 2};
    double a_data[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    std::vector<size_t> b_shape = {2, 3, 2, 2};
    double b_data[24] = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                         113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124};

    tla::Tensor A = tla::Tensor::tensor(a_data, a_shape, tla::Device::CUDA);
    tla::Tensor B = tla::Tensor::tensor(b_data, b_shape, tla::Device::CUDA);

    tla::Tensor C = A.matmul(B);

    assert(C.ndim() == 4);
    assert(C.numel() == 24);
    assert(C.device() == tla::Device::CUDA);

    double expected[24] = {307,  310,  715,  722,  1167, 1178, 1591, 1606, 2091, 2110, 2531, 2554,
                           3079, 3106, 3535, 3566, 4131, 4166, 4603, 4642, 5247, 5290, 5735, 5782};

    tla::Tensor host_c = C.to_cpu();
    const double* c_data = host_c.data();
    for (size_t i = 0; i < 24; i++) {
        assert(c_data[i] == expected[i]);
    }

    printf("✓ CUDA matmul (4D batched) passed\n");
}

int main() {
    tinyla_init();

    printf("Running GPU tests\n================\n");
    test_tensor_creation_gpu();
    test_tensor_print_gpu();
    test_add_gpu();
    test_matmul_cuda_2d();
    test_matmul_cuda_3d();
    test_matmul_cuda_4d();
    printf("\nAll GPU tests passed!\n");
    return 0;
}
