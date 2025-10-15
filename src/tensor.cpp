#include "tinyla/tensor.hpp"
#include "backend/op_wrapper.h"
#include "memory/tensor_desc.h"
#include "rand/rand.h"

// Internal constructor
tla::Tensor::Tensor(const std::vector<size_t> shape, tla::Device device, buffer_init_mode init_mode)
    : desc_(tensor_desc_create(NULL, shape.data(), shape.size(), static_cast<::device>(device), init_mode)) {}
tla::Tensor::Tensor(const double* data, const std::vector<size_t> shape, tla::Device device)
    : desc_(tensor_desc_create(data, shape.data(), shape.size(), static_cast<::device>(device),
                               buffer_init_mode::FROM_DATA)) {}
tla::Tensor::Tensor(tensor_desc* desc) : desc_(desc) {}

// Destructor
tla::Tensor::~Tensor() { tensor_desc_free(&desc_); }

// Move constructor
tla::Tensor::Tensor(Tensor&& other) noexcept : desc_(other.desc_) { other.desc_ = nullptr; }

// Move assignment
tla::Tensor& tla::Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        tensor_desc_free(&desc_);
        desc_ = other.desc_;
        other.desc_ = nullptr;
    }
    return *this;
}

// Public constructors
tla::Tensor tla::Tensor::tensor(const double* data, const std::vector<size_t> shape, tla::Device device) {
    return Tensor(data, shape, device);
}
tla::Tensor tla::Tensor::rand(const std::vector<size_t> shape, tla::Device device, uint64_t seed) {
    return Tensor(rand_tensor(shape.data(), shape.size(), static_cast<::device>(device), seed));
}
tla::Tensor tla::Tensor::randn(const std::vector<size_t> shape, tla::Device device, uint64_t seed) {
    return Tensor(randn_tensor(shape.data(), shape.size(), static_cast<::device>(device), seed));
}
tla::Tensor tla::Tensor::rand_like(const Tensor& other, tla::Device device, uint64_t seed) {
    return Tensor(rand_tensor_like(other.desc_, static_cast<::device>(device), seed));
}
tla::Tensor tla::Tensor::randn_like(const Tensor& other, tla::Device device, uint64_t seed) {
    return Tensor(randn_tensor_like(other.desc_, static_cast<::device>(device), seed));
}
tla::Tensor tla::Tensor::empty(const std::vector<size_t> shape, tla::Device device) {
    return Tensor(shape, device, buffer_init_mode::UNINITIALIZED);
}
tla::Tensor tla::Tensor::empty_like(const Tensor& other) {
    return Tensor(other.shape(), other.device(), buffer_init_mode::UNINITIALIZED);
}
tla::Tensor tla::Tensor::zeroes(const std::vector<size_t> shape, tla::Device device) {
    return Tensor(shape, device, buffer_init_mode::ZEROS);
}
tla::Tensor tla::Tensor::zeroes_like(const Tensor& other) {
    return Tensor(other.shape(), other.device(), buffer_init_mode::ZEROS);
}
tla::Tensor tla::Tensor::ones(const std::vector<size_t> shape, tla::Device device) {
    return Tensor(shape, device, buffer_init_mode::ONES);
}
tla::Tensor tla::Tensor::ones_like(const Tensor& other) {
    return Tensor(other.shape(), other.device(), buffer_init_mode::ONES);
}

// Getters
size_t tla::Tensor::ndim() const { return desc_->ndim; }
size_t tla::Tensor::numel() const { return desc_->numel; }
tla::Device tla::Tensor::device() const { return static_cast<tla::Device>(desc_->device); }
const std::vector<size_t> tla::Tensor::shape() const {
    return std::vector<size_t>(desc_->shape, desc_->shape + desc_->ndim);
}

double* tla::Tensor::data() { return desc_->buffer; }
const double* tla::Tensor::data() const { return desc_->buffer; }

// Device conversion
tla::Tensor tla::Tensor::to(tla::Device device, bool force_copy) const {
    return Tensor(tensor_desc_to_device(desc_, static_cast<::device>(device), force_copy));
}
tla::Tensor tla::Tensor::to_cpu() const { return to(tla::Device::CPU, false); }
tla::Tensor tla::Tensor::to_cuda() const { return to(tla::Device::CUDA, false); }

// Printing
void tla::Tensor::print() const { tensor_desc_print(desc_); }

// Operations
tla::Tensor tla::Tensor::add(const Tensor& other) const { return execute_binary_op(other, add_op_wrapper, "add"); }
tla::Tensor tla::Tensor::sub(const Tensor& other) const { return execute_binary_op(other, sub_op_wrapper, "sub"); }
tla::Tensor tla::Tensor::mul(const Tensor& other) const { return execute_binary_op(other, mul_op_wrapper, "mul"); }
tla::Tensor tla::Tensor::div(const Tensor& other) const { return execute_binary_op(other, div_op_wrapper, "div"); }
tla::Tensor tla::Tensor::pow(const Tensor& other) const { return execute_binary_op(other, pow_op_wrapper, "pow"); }
tla::Tensor tla::Tensor::matmul(const Tensor& other) const {
    return execute_binary_op(other, matmul_op_wrapper, "matmul");
}
tla::Tensor tla::Tensor::operator+(const Tensor& other) const { return add(other); }
tla::Tensor tla::Tensor::operator-(const Tensor& other) const { return sub(other); }
tla::Tensor tla::Tensor::operator*(const Tensor& other) const { return mul(other); }
tla::Tensor tla::Tensor::operator/(const Tensor& other) const { return div(other); }
tla::Tensor tla::Tensor::operator^(const Tensor& other) const { return pow(other); }
