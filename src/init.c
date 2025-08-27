#include "tinyla/init.h"
#include "backend/dispatcher.h"
#include "cpu/ops/add_scalar.h"

#ifdef TINYLA_CUDA_ENABLED
#include "cuda/ops/add.h"
#endif

void tinyla_init(void) {
#ifdef TINYLA_CUDA_ENABLED
    register_op(OP_ADD, OP_ARITY_BINARY, cpu_add_scalar, gpu_add_kernel);
#else
    register_op(OP_ADD, OP_ARITY_BINARY, cpu_add_scalar, NULL);
#endif
}