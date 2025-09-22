#include "../../cuda_macros.h"
#include "gemm.h"
#include <cuda_runtime.h>

__global__ void gemm_kernel(double* out, const double* a, const double* b, size_t m, size_t k, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (size_t k_idx = 0; k_idx < k; k_idx++) {
            sum += a[row * k + k_idx] * b[k_idx * n + col];
        }
        out[row * n + col] = sum;
    }
}
__global__ void gemm_batched_kernel(double* out, const double* a, const double* b, size_t m, size_t k, size_t n,
                                    size_t batch_count) {
    size_t batch = blockIdx.z;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_count || row >= m || col >= n) {
        return;
    }

    double sum = 0.0;
    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        sum += a[batch * m * k + row * k + k_idx] * b[batch * k * n + k_idx * n + col];
    }
    out[batch * m * n + row * n + col] = sum;
}

int cuda_gemm(Tensor* out, const Tensor** inputs, const size_t n_inputs) {
    const Tensor* a = inputs[0];
    const Tensor* b = inputs[1];

    GemmDims dims = extract_gemm_dims(a, b);
    const size_t batch_count = dims.batch_dim ? dims.batch_dim : 1;
    dim3 block(16, 16, 1);
    dim3 grid((dims.m + block.x - 1) / block.x, (dims.n + block.y - 1) / block.y, batch_count);

    if (batch_count == 1) {
        gemm_kernel<<<grid, block>>>(out->buffer, a->buffer, b->buffer, dims.m, dims.k, dims.n);
    } else {
        gemm_batched_kernel<<<grid, block>>>(out->buffer, a->buffer, b->buffer, dims.m, dims.k, dims.n, batch_count);
    }
    TLA_CUDA_KERNEL_CHECK();

    return 0;
}