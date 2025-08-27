#include <stddef.h>
#include <stdio.h>

int cpu_add_scalar(double** inputs, double* out, size_t numel) {
    if (!inputs || !out) {
        fprintf(stderr, "cpu_add_scalar: NULL pointer\n");
        return -1;
    }

    double* a = inputs[0];
    double* b = inputs[1];
    if (!a || !b) {
        fprintf(stderr, "cpu_add_scalar: NULL input buffer\n");
        return -1;
    }

    if (numel == 0) {
        return 0;
    }

    for (size_t i = 0; i < numel; i++) {
        out[i] = a[i] + b[i];
    }

    return 0;
}
