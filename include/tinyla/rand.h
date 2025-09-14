#pragma once
#include "tensor.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a tensor filled with random values from a uniform distribution.
 *
 * This function creates a new tensor and initializes all elements with random values
 * drawn from a uniform distribution over the interval [0.0, 1.0). The random number
 * generation uses the Philox algorithm for high-quality pseudorandom numbers.
 * The tensor is allocated on the specified device and all elements are randomly initialized.
 *
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 * @param seed Seed value for the random number generator. Used to ensure reproducible
 *             random sequences. The same seed will produce the same sequence of values.
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         All elements in the returned tensor are initialized with random values
 *         from a uniform distribution over [0.0, 1.0).
 */
Tensor* rand_tensor(const size_t* shape, size_t ndim, Device device, uint64_t seed);

/**
 * @brief Creates a tensor filled with random values from a normal distribution.
 *
 * This function creates a new tensor and initializes all elements with random values
 * drawn from a normal (Gaussian) distribution with mean 0.0 and standard deviation 1.0.
 * The random number generation uses the Philox algorithm for high-quality pseudorandom
 * numbers, combined with the Box-Muller transform for normal distribution sampling.
 * The tensor is allocated on the specified device and all elements are randomly initialized.
 *
 * @param shape Array specifying the dimensions of the tensor. Must not be NULL.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 * @param seed Seed value for the random number generator. Used to ensure reproducible
 *             random sequences. The same seed will produce the same sequence of values.
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         All elements in the returned tensor are initialized with random values
 *         from a normal distribution with mean 0.0 and standard deviation 1.0.
 */
Tensor* randn_tensor(const size_t* shape, size_t ndim, Device device, uint64_t seed);

/**
 * @brief Creates a tensor filled with random values from a uniform distribution, matching another tensor.
 *
 * This function creates a new tensor with the same shape and device as the input tensor,
 * but initializes all elements with random values from a uniform distribution over [0.0, 1.0).
 * The random number generation uses the Philox algorithm for high-quality pseudorandom numbers.
 *
 * @param tensor Reference tensor to match shape and device. Must not be NULL.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 * @param seed Seed value for the random number generator. Used to ensure reproducible
 *             random sequences. The same seed will produce the same sequence of values.
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         The returned tensor has the same shape and device as the input tensor.
 *         All elements in the returned tensor are initialized with random values
 *         from a uniform distribution over [0.0, 1.0).
 */
Tensor* rand_tensor_like(const Tensor* tensor, Device device, uint64_t seed);

/**
 * @brief Creates a tensor filled with random values from a normal distribution, matching another tensor.
 *
 * This function creates a new tensor with the same shape and device as the input tensor,
 * but initializes all elements with random values from a normal distribution with mean 0.0
 * and standard deviation 1.0. The random number generation uses the Philox algorithm for
 * high-quality pseudorandom numbers, combined with the Box-Muller transform for normal
 * distribution sampling.
 *
 * @param tensor Reference tensor to match shape and device. Must not be NULL.
 * @param device Target device for tensor storage (DEVICE_CPU or DEVICE_GPU).
 * @param seed Seed value for the random number generator. Used to ensure reproducible
 *             random sequences. The same seed will produce the same sequence of values.
 *
 * @return Pointer to the created tensor on success, NULL on failure.
 *         The returned tensor has the same shape and device as the input tensor.
 *         All elements in the returned tensor are initialized with random values
 *         from a normal distribution with mean 0.0 and standard deviation 1.0.
 */
Tensor* randn_tensor_like(const Tensor* tensor, Device device, uint64_t seed);

#ifdef __cplusplus
}
#endif