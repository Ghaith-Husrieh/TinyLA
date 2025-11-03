#pragma once
#include <cuda_runtime.h>

struct AddOp {
    __device__ static double apply(double a, double b) { return a + b; }
};

struct SubOp {
    __device__ static double apply(double a, double b) { return a - b; }
};

struct MulOp {
    __device__ static double apply(double a, double b) { return a * b; }
};

struct DivOp {
    __device__ static double apply(double a, double b) { return a / b; }
};

struct PowOp {
    __device__ static double apply(double a, double b) {
        if (b == 0.0)
            return 1.0;
        if (b == 1.0)
            return a;
        if (b == -1.0)
            return 1.0 / a;
        if (b == 2.0)
            return a * a;
        if (b == 3.0)
            return a * a * a;
        if (b == -2.0)
            return 1.0 / (a * a);
        return pow(a, b);
    }
};
