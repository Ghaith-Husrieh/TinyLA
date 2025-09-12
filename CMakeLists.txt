cmake_minimum_required(VERSION 4.1.0)
project(TinyLA VERSION 0.1.0 LANGUAGES C CUDA)

# ===========================
# Compiler Standards
# ===========================
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Remove DLL/lib prefixes on Windows (cleaner names)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")

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
)

# ===========================
# OpenMP Support
# ===========================
find_package(OpenMP REQUIRED COMPONENTS C)
message(STATUS "OpenMP detected")

# ===========================
# CPU Kernels
# ===========================
file(GLOB_RECURSE SCALAR_KERNELS src/cpu/ops/scalar/*.c)
file(GLOB_RECURSE VEC128_KERNELS src/cpu/ops/vec128/*.c)
file(GLOB_RECURSE VEC256_KERNELS src/cpu/ops/vec256/*.c)

# Create object libraries dynamically
add_library(cpu_scalar_objs OBJECT ${SCALAR_KERNELS})
add_library(cpu_vec128_objs OBJECT ${VEC128_KERNELS})
add_library(cpu_vec256_objs OBJECT ${VEC256_KERNELS})

# Compile flags per ISA
if(MSVC)
    target_compile_options(cpu_vec128_objs PRIVATE /arch:SSE2)
    target_compile_options(cpu_vec256_objs PRIVATE /arch:AVX2)
else()
    target_compile_options(cpu_vec128_objs PRIVATE -msse4.2)
    target_compile_options(cpu_vec256_objs PRIVATE -mavx2)
endif()

# ===========================
# CUDA Support (Optional)
# ===========================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    file(GLOB_RECURSE CUDA_KERNELS src/cuda/ops/*.cu)
    file(GLOB_RECURSE CUDA_INIT src/cuda/init/*.cu)

    set(CUDA_SOURCES ${CUDA_INIT} ${CUDA_KERNELS})

    message(STATUS "CUDAToolkit ${CUDAToolkit_VERSION} detected - GPU acceleration enabled")
else()
    set(CUDA_SOURCES "")
    message(STATUS "CUDAToolkit not detected - GPU acceleration disabled")
endif()

# ===========================
# Build TinyLA Shared Library
# ===========================
add_library(TinyLA SHARED
    $<TARGET_OBJECTS:cpu_scalar_objs>
    $<TARGET_OBJECTS:cpu_vec128_objs>
    $<TARGET_OBJECTS:cpu_vec256_objs>
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

# Include headers
target_include_directories(TinyLA PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Enable CUDA features if available
if(CUDAToolkit_FOUND)
    target_compile_definitions(TinyLA PRIVATE TINYLA_CUDA_ENABLED)
    target_link_libraries(TinyLA PRIVATE CUDA::cudart)
endif()

# Enable OpenMP features
if(MSVC)
    target_compile_options(cpu_scalar_objs  PRIVATE /openmp)
    target_compile_options(cpu_vec128_objs  PRIVATE /openmp)
    target_compile_options(cpu_vec256_objs  PRIVATE /openmp)
else()
    target_compile_options(cpu_scalar_objs  PRIVATE -fopenmp)
    target_compile_options(cpu_vec128_objs  PRIVATE -fopenmp)
    target_compile_options(cpu_vec256_objs  PRIVATE -fopenmp)

    target_link_libraries(TinyLA PRIVATE OpenMP::OpenMP_C)
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
