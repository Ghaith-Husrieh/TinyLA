#pragma once
#include <stddef.h>

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU,
} Device;

struct ndarray {
    size_t ndim;
    size_t numel;
    size_t* shape;
    double* buffer;
    Device device;

    void (*free)(struct ndarray**);
    void (*print)(const struct ndarray*);
};

typedef struct ndarray Tensor;

/**
 * @brief Move a tensor to the specified device.
 *
 * If the tensor is already on the target device and `force_copy` is 0, it is returned as-is.
 * Otherwise, a new tensor is created on the target device and data is copied.
 *
 * @param tensor Pointer to the tensor to move.
 * @param device Target device (DEVICE_CPU or DEVICE_GPU).
 * @param force_copy If non-zero, always create a new tensor even if already on target device.
 *
 * @return Pointer to the tensor on the requested device.
 */
Tensor* to(const Tensor* tensor, Device device, int force_copy);

/**
 * @brief Move a tensor to CPU.
 *
 * Convenience macro for to(tensor, DEVICE_CPU, 0).
 */
#define to_cpu(tensor) to(tensor, DEVICE_CPU, 0)

/**
 * @brief Move a tensor to GPU.
 *
 * Convenience macro for to(tensor, DEVICE_GPU, 0).
 */
#define to_gpu(tensor) to(tensor, DEVICE_GPU, 0)

/**
 * @brief Creates a tensor with the provided data.
 *
 * This function creates a new tensor and initializes it with the provided data.
 * The data is copied to the specified device (CPU or GPU) and the tensor
 * structure is properly initialized with shape and metadata.
 *
 * @param data Pointer to the input data array. Must not be NULL.
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         Failure can occur due to invalid parameters, memory allocation
 *         failure, or GPU memory allocation failure (for DEVICE_GPU).
 *
 * @note The function copies the input data to the target device. For GPU tensors,
 *       the data is transferred from CPU to GPU memory.
 */
Tensor* tensor(const double* data, const size_t* shape, size_t ndim, Device device);

/**
 * @brief Creates an uninitialized tensor.
 *
 * This function creates a new tensor with allocated memory but leaves the
 * contents uninitialized. This is useful when you plan to fill the tensor
 * with data later or when you want to avoid the overhead of initialization.
 *
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         The tensor memory contains undefined values and should be
 *         initialized before use.
 *
 * @note This is the fastest tensor creation method as it skips initialization.
 *       The user is responsible for initializing the tensor contents.
 */
Tensor* empty_tensor(const size_t* shape, size_t ndim, Device device);

/**
 * @brief Creates a tensor filled with zeros.
 *
 * This function creates a new tensor and initializes all elements to zero.
 * The tensor is allocated on the specified device and all memory is set to 0.
 *
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         All elements in the returned tensor are initialized to 0.0.
 *
 * @note For GPU tensors, the zero initialization is performed on the GPU
 *       using cudaMemset for efficiency.
 */
Tensor* zeroes_tensor(const size_t* shape, size_t ndim, Device device);

/**
 * @brief Creates a tensor filled with ones.
 *
 * This function creates a new tensor and initializes all elements to 1.0.
 * The tensor is allocated on the specified device and all memory is set to 1.0.
 *
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         All elements in the returned tensor are initialized to 1.0.
 *
 * @note For GPU tensors, the initialization is performed using a CUDA kernel
 *       for efficiency. For CPU tensors, a simple loop is used.
 */
Tensor* ones_tensor(const size_t* shape, size_t ndim, Device device);
