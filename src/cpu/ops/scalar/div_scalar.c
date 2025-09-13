#include "../div.h"
#include <omp.h>

int cpu_div_scalar(const double** inputs, double* out, size_t numel) {
    int i;

#pragma omp parallel for
    for (i = 0; i < numel; i++) {
        out[i] = inputs[0][i] / inputs[1][i];
    }

    return 0;
}