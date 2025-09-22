cmake_minimum_required(VERSION 4.1.0)
project(TinyLA VERSION 0.1.0 LANGUAGES C CXX CUDA)

# ===========================
# Compiler Standards
# ===========================
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Remove DLL/lib prefixes on Windows (cleaner names)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")

# ===========================
# Compiler Features
# ===========================

if (NOT MSVC)
    include(CheckSymbolExists)
    check_symbol_exists(posix_memalign "stdlib.h" HAVE_POSIX_MEMALIGN)
endif()

# ===========================
# Core Sources
# ===========================
set(CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/tensor.c
    ${PROJECT_SOURCE_DIR}/src/init.c
    ${PROJECT_SOURCE_DIR}/src/memory/tla_alloc.c
    ${PROJECT_SOURCE_DIR}/src/backend/dispatcher.c
    ${PROJECT_SOURCE_DIR}/src/backend/op_wrapper.c
    ${PROJECT_SOURCE_DIR}/src/cpu/cpu_features.c
    ${PROJECT_SOURCE_DIR}/src/cpu/alignment.c
    ${PROJECT_SOURCE_DIR}/src/rand/rand.cpp
)

file(GLOB_RECURSE CPU_KERNELS 
    ${PROJECT_SOURCE_DIR}/src/cpu/ops/element_wise/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/cpu/ops/gemm/*.cpp
)

set(CORE_SOURCES ${CORE_SOURCES} ${CPU_KERNELS})

# ===========================
# OpenMP Support
# ===========================
find_package(OpenMP REQUIRED COMPONENTS C CXX)
message(STATUS "OpenMP detected")

# ===========================
# CUDA Support (Optional)
# ===========================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    file(GLOB_RECURSE CUDA_KERNELS 
    src/cuda/ops/element_wise/*.cu
    src/cuda/ops/gemm/*.cu
    )
    file(GLOB_RECURSE CUDA_INIT src/cuda/init/*.cu)
    file(GLOB_RECURSE CUDA_RAND src/rand/*.cu)

    set(CUDA_SOURCES ${CUDA_INIT} ${CUDA_KERNELS} ${CUDA_RAND})

    message(STATUS "CUDAToolkit ${CUDAToolkit_VERSION} detected - GPU acceleration enabled")
else()
    set(CUDA_SOURCES "")
    message(STATUS "CUDAToolkit not detected - GPU acceleration disabled")
endif()

# ===========================
# Build TinyLA Shared Library
# ===========================
add_library(TinyLA SHARED
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

if (HAVE_POSIX_MEMALIGN)
target_compile_definitions(TinyLA PRIVATE HAVE_POSIX_MEMALIGN)
endif()

# Include headers
target_include_directories(TinyLA PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(TinyLA PUBLIC ${PROJECT_SOURCE_DIR}/src/)

# Enable CUDA features if available
if(CUDAToolkit_FOUND)
    target_compile_definitions(TinyLA PRIVATE TINYLA_CUDA_ENABLED)
    target_link_libraries(TinyLA PRIVATE CUDA::cudart)
endif()

# Enable OpenMP features
if(MSVC)
    target_compile_options(TinyLA PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>
        $<$<COMPILE_LANGUAGE:C>:/openmp:llvm>
    )
else()
    target_compile_options(TinyLA PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
        $<$<COMPILE_LANGUAGE:C>:-fopenmp>
    )
    target_link_libraries(TinyLA PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()


# Windows-specific: export all symbols
set_target_properties(TinyLA PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# ===========================
# Testing
# ===========================
enable_testing()

# CPU tests
file(GLOB CPU_TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/test_cpu.c)
foreach(test_src ${CPU_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${test_name} PRIVATE TinyLA)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# GPU tests (only if CUDA is available)
if(CUDAToolkit_FOUND)
    file(GLOB GPU_TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/test_cuda.c)
    foreach(test_src ${GPU_TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${test_name} PRIVATE TinyLA CUDA::cudart)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()