<h1 align="center"> TinyLA </h1>

A lightweight linear algebra library for C++ with CPU and CUDA GPU support.

> [!Note]
>
> - Currently only double precision (64-bit) is supported.

## Usage

```cpp
#include "tinyla/tinyla.h"

tinyla_init();

// Create tensors
auto a = tla::Tensor::rand({16, 16}, tla::Device::CPU);
auto b = tla::Tensor::ones({16, 16}, tla::Device::CPU);

// Operations
auto c = a + b;           // Element-wise addition
auto d = a.matmul(b);     // Matrix multiplication

// Print results
c.print();
d.print();

// GPU support (if CUDA available)
auto gpu_a = a.to_cuda();
```

## Requirements

- C/C++17
- CMake 4.1.0+
- OpenMP
- CUDA Toolkit (required for GPU acceleration)
