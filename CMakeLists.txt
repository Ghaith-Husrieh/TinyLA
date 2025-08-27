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
set(LIB_SOURCES
    ${PROJECT_SOURCE_DIR}/src/tensor.c
    ${PROJECT_SOURCE_DIR}/src/init.c
    ${PROJECT_SOURCE_DIR}/src/memory/tla_alloc.c
    ${PROJECT_SOURCE_DIR}/src/backend/dispatcher.c
    ${PROJECT_SOURCE_DIR}/src/backend/op_wrapper.c
    ${PROJECT_SOURCE_DIR}/src/cpu/ops/add_scalar.c
)

# ===========================
# CUDA Support (Optional)
# ===========================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    list(APPEND LIB_SOURCES
        ${PROJECT_SOURCE_DIR}/src/cuda/init/ones.cu
        ${PROJECT_SOURCE_DIR}/src/cuda/ops/add.cu
    )

    message(STATUS "CUDAToolkit ${CUDAToolkit_VERSION} detected - GPU acceleration enabled")
else()
    message(STATUS "CUDAToolkit not detected - GPU acceleration disabled")
endif()

# ===========================
# Build TinyLA Shared Library
# ===========================
add_library(TinyLA SHARED ${LIB_SOURCES})

# Include headers
target_include_directories(TinyLA PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Enable CUDA features if available
if(CUDAToolkit_FOUND)
    target_compile_definitions(TinyLA PRIVATE TINYLA_CUDA_ENABLED)
    target_link_libraries(TinyLA PRIVATE CUDA::cudart)
endif()

# Windows-specific: export all symbols
set_target_properties(TinyLA PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

#===============================
# Testing
#===============================
# Enable testing
enable_testing()

# Add tests directory
file(GLOB CPU_TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/test_cpu.c)

# CPU tests
foreach(test_src ${CPU_TEST_SOURCES})
    # Extract the filename without extension
    get_filename_component(test_name ${test_src} NAME_WE)

    # Create executable for each test
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${test_name} PRIVATE TinyLA)

    # Register test with CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# GPU tests (only if CUDA is available)
if(CUDAToolkit_FOUND)
    file(GLOB GPU_TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/test_cuda.c)

    foreach(test_src ${GPU_TEST_SOURCES})
        # Extract the filename without extension
        get_filename_component(test_name ${test_src} NAME_WE)

        # Create executable for each test
        add_executable(${test_name} ${test_src})
        target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${test_name} PRIVATE TinyLA CUDA::cudart)

        # Register test with CTest
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()