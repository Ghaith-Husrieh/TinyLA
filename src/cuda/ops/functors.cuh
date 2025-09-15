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