#pragma once
#include "memory/tensor_desc.h"
#include <cstddef>
#include <vector>

namespace tla {

/**
 * @brief Device types supported by TinyLA
 */
enum class Device {
    CPU,  ///< CPU computation
    CUDA, ///< CUDA GPU computation
};

/**
 * @brief A multi-dimensional array with automatic memory management
 *
 * The Tensor class provides a modern C++ interface for multi-dimensional arrays
 * with support for both CPU and CUDA computation. Memory is automatically managed
 * using RAII (Resource Acquisition Is Initialization).
 */
class Tensor {
  public:
    /**
     * @brief Destructor - automatically frees memory
     */
    ~Tensor();

    /**
     * @brief Move constructor
     * @param other The tensor to move from
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief Move assignment operator
     * @param other The tensor to move from
     * @return Reference to this tensor
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief Copy constructor (deleted - use move semantics)
     */
    Tensor(const Tensor&) = delete;

    /**
     * @brief Copy assignment operator (deleted - use move semantics)
     */
    Tensor& operator=(const Tensor&) = delete;

    // ===========================
    // Factory Methods
    // ===========================

    /**
     * @brief Create a tensor from existing data
     * @param data Pointer to the data array
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @return A new tensor with the provided data
     */
    static Tensor tensor(const double* data, const std::vector<size_t> shape, Device device);

    /**
     * @brief Create a tensor with random uniform values [0, 1)
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @param seed Random seed (0 for random seed)
     * @return A new tensor with random uniform values
     */
    static Tensor rand(const std::vector<size_t> shape, Device device, uint64_t seed = 0);

    /**
     * @brief Create a tensor with random normal values (mean=0, std=1)
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @param seed Random seed (0 for random seed)
     * @return A new tensor with random normal values
     */
    static Tensor randn(const std::vector<size_t> shape, Device device, uint64_t seed = 0);

    /**
     * @brief Create a random tensor with the same shape as another tensor
     * @param other The tensor to match the shape of
     * @param device The device to create the tensor on
     * @param seed Random seed (0 for random seed)
     * @return A new tensor with random uniform values
     */
    static Tensor rand_like(const Tensor& other, Device device, uint64_t seed = 0);

    /**
     * @brief Create a random normal tensor with the same shape as another tensor
     * @param other The tensor to match the shape of
     * @param device The device to create the tensor on
     * @param seed Random seed (0 for random seed)
     * @return A new tensor with random normal values
     */
    static Tensor randn_like(const Tensor& other, Device device, uint64_t seed = 0);

    /**
     * @brief Create an uninitialized tensor
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @return A new uninitialized tensor
     */
    static Tensor empty(const std::vector<size_t> shape, Device device);

    /**
     * @brief Create an uninitialized tensor with the same shape as another tensor
     * @param other The tensor to match the shape of
     * @return A new uninitialized tensor
     */
    static Tensor empty_like(const Tensor& other);

    /**
     * @brief Create a tensor filled with zeros
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @return A new tensor filled with zeros
     */
    static Tensor zeroes(const std::vector<size_t> shape, Device device);

    /**
     * @brief Create a tensor filled with zeros with the same shape as another tensor
     * @param other The tensor to match the shape of
     * @return A new tensor filled with zeros
     */
    static Tensor zeroes_like(const Tensor& other);

    /**
     * @brief Create a tensor filled with ones
     * @param shape The shape of the tensor
     * @param device The device to create the tensor on
     * @return A new tensor filled with ones
     */
    static Tensor ones(const std::vector<size_t> shape, Device device);

    /**
     * @brief Create a tensor filled with ones with the same shape as another tensor
     * @param other The tensor to match the shape of
     * @return A new tensor filled with ones
     */
    static Tensor ones_like(const Tensor& other);

    // ===========================
    // Accessors
    // ===========================

    /**
     * @brief Get the number of dimensions
     * @return The number of dimensions
     */
    size_t ndim() const;

    /**
     * @brief Get the total number of elements
     * @return The total number of elements
     */
    size_t numel() const;

    /**
     * @brief Get the device this tensor is on
     * @return The device type
     */
    Device device() const;

    /**
     * @brief Get the shape of the tensor
     * @return A vector containing the dimensions
     */
    const std::vector<size_t> shape() const;

    /**
     * @brief Get a pointer to the underlying data
     * @return Pointer to the data (mutable)
     */
    double* data();

    /**
     * @brief Get a pointer to the underlying data
     * @return Pointer to the data (const)
     */
    const double* data() const;

    // ===========================
    // Device Operations
    // ===========================

    /**
     * @brief Move tensor to a different device
     * @param device The target device
     * @param force_copy Force a copy even if already on the target device
     * @return A new tensor on the target device
     */
    Tensor to(Device device, bool force_copy = false) const;

    /**
     * @brief Move tensor to CPU
     * @return A new tensor on CPU
     */
    Tensor to_cpu() const;

    /**
     * @brief Move tensor to CUDA
     * @return A new tensor on CUDA
     */
    Tensor to_cuda() const;

    /**
     * @brief Print the tensor to stdout
     */
    void print() const;

    // ===========================
    // Operations
    // ===========================

    /**
     * @brief Element-wise addition
     * @param other The tensor to add
     * @return A new tensor with the result
     */
    Tensor add(const Tensor& other) const;

    /**
     * @brief Element-wise subtraction
     * @param other The tensor to subtract
     * @return A new tensor with the result
     */
    Tensor sub(const Tensor& other) const;

    /**
     * @brief Element-wise multiplication
     * @param other The tensor to multiply with
     * @return A new tensor with the result
     */
    Tensor mul(const Tensor& other) const;

    /**
     * @brief Element-wise division
     * @param other The tensor to divide by
     * @return A new tensor with the result
     */
    Tensor div(const Tensor& other) const;

    /**
     * @brief Element-wise power operation
     * @param other The tensor containing the exponents
     * @return A new tensor with the result
     */
    Tensor pow(const Tensor& other) const;

    /**
     * @brief Matrix multiplication
     * @param other The tensor to multiply with
     * @return A new tensor with the result
     */
    Tensor matmul(const Tensor& other) const;

    // ===========================
    // Operator Overloading
    // ===========================

    /**
     * @brief Element-wise addition operator
     * @param other The tensor to add
     * @return A new tensor with the result
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Element-wise subtraction operator
     * @param other The tensor to subtract
     * @return A new tensor with the result
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Element-wise multiplication operator
     * @param other The tensor to multiply with
     * @return A new tensor with the result
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Element-wise division operator
     * @param other The tensor to divide by
     * @return A new tensor with the result
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * @brief Element-wise power operator
     * @param other The tensor containing the exponents
     * @return A new tensor with the result
     */
    Tensor operator^(const Tensor& other) const;

  private:
    Tensor(const std::vector<size_t> shape,
           Device device,
           buffer_init_mode init_mode = buffer_init_mode::UNINITIALIZED);
    Tensor(const double* data, const std::vector<size_t> shape, Device device);
    Tensor(tensor_desc* desc);

    template <typename OpWrapper>
    Tensor execute_binary_op(const Tensor& other, OpWrapper op_func, const char* op_name) const {
        tensor_desc* out =
                tensor_desc_create(NULL, desc_->shape, desc_->ndim, desc_->device, buffer_init_mode::UNINITIALIZED);
        if (!out) {
            fprintf(stderr, "Failed to allocate output tensor for %s\n", op_name);
            return Tensor(nullptr);
        }

        if (op_func(out, desc_, other.desc_) != 0) {
            fprintf(stderr, "Failed to execute %s operation\n", op_name);
            tensor_desc_free(&out);
            return Tensor(nullptr);
        }

        return Tensor(out);
    }

    tensor_desc* desc_ = nullptr;
};

} // namespace tla
