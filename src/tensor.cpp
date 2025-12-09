#include "tinyla/tensor.hpp"
#include "memory/tensor_desc.h"
#include "rand/rand.h"

// Internal constructor
tla::Tensor::Tensor(const std::vector<size_t> shape, tla::Device device, buffer_init_mode init_mode)
    : desc_(tensor_desc_create(NULL, shape.data(), shape.size(), static_cast<::device>(device), init_mode)) {}
tla::Tensor::Tensor(const double* data, const std::vector<size_t> shape, tla::Device device)
    : desc_(tensor_desc_create(data,
                               shape.data(),
                               shape.size(),
                               static_cast<::device>(device),
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
tla::Tensor tla::Tensor::to(tla::Device device) const {
    return Tensor(tensor_desc_to_device(desc_, static_cast<::device>(device)));
}
tla::Tensor tla::Tensor::to_cpu() const { return to(tla::Device::CPU); }
tla::Tensor tla::Tensor::to_cuda() const { return to(tla::Device::CUDA); }

// Printing
void tla::Tensor::print() const { tensor_desc_print(desc_); }

// Execute operations
tla::Tensor tla::Tensor::infer_output_shape(OpKind kind, const tensor_desc** inputs, size_t n_inputs) const {
    std::vector<size_t> output_shape;

    switch (kind) {
    case OP_ELEMENT_WISE:
        output_shape = std::vector<size_t>(inputs[0]->shape, inputs[0]->shape + inputs[0]->ndim);
        break;
    case OP_GEMM:
        for (size_t i = 0; i < inputs[0]->ndim - 2; i++) {
            output_shape.push_back(inputs[0]->shape[i]);
        }
        output_shape.push_back(inputs[0]->shape[inputs[0]->ndim - 2]);
        output_shape.push_back(inputs[1]->shape[inputs[1]->ndim - 1]);
        break;
    default:
        fprintf(stderr, "Invalid operation kind: %d.\n", static_cast<int>(kind));
        return Tensor(nullptr);
    }

    return empty(output_shape, static_cast<tla::Device>(inputs[0]->device));
}

tla::Tensor tla::Tensor::execute_op(OpType op, const tensor_desc** inputs, size_t n_inputs) const {
    OpEntry* entry = get_op_entry(op);
    if (!entry) {
        fprintf(stderr, "Invalid operation: %d.\n", op);
        return Tensor(nullptr);
    }

    if (entry->arity != n_inputs) {
        fprintf(stderr,
                "Arity mismatch for %s operation: expected %d, got %zu.\n",
                entry->verbose_name,
                entry->arity,
                n_inputs);
        return Tensor(nullptr);
    }

    if (!entry->validator) {
        fprintf(stderr, "No validator registered for %s operation.\n", entry->verbose_name);
        return Tensor(nullptr);
    }

    if (entry->validator(inputs, n_inputs) != 0) {
        fprintf(stderr, "Validation failed for %s operation.\n", entry->verbose_name);
        return Tensor(nullptr);
    }

    Tensor out = infer_output_shape(entry->kind, inputs, n_inputs);
    if (dispatch_kernel(entry, inputs, n_inputs, out.desc_) != 0) {
        fprintf(stderr, "Failed to execute %s operation.\n", entry->verbose_name);
        return Tensor(nullptr);
    }
    return out;
}
