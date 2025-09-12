#include "../add.h"
#include <stddef.h>
#include <stdio.h>

int cpu_add_scalar(const double** inputs, double* out, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        out[i] = inputs[0][i] + inputs[1][i];
    }

    return 0;
}
