#pragma once

/**
 * @brief Initializes TinyLA operation kernels.
 *
 * This function registers CPU and GPU kernels for supported operations.
 * It must be called before calling any operations (e.g., add) that rely
 * on these kernels. Tensor creation functions do not require this.
 */
void tinyla_init(void);