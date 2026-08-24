cmake_minimum_required(VERSION 3.17)
project(GpuHnsw LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

option(PROFILE_BUILD_PHASES "Instrument build_graph_kernel per-phase GPU cycle counters" OFF)
if(PROFILE_BUILD_PHASES)
  add_compile_definitions(PROFILE_BUILD_PHASES=1)
endif()
# cmake_policy(SET CMP0074 NEW)
# enable_language(CUDA)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -G for device debugging
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 --ptxas-options=-v")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
set(CUDAToolkit_ROOT "/usr/local/cuda-12.3/")
# "native" can fail under some environments (e.g. WSL/sandboxed builds) and
# silently fall back to an older default arch that does not support WMMA.
# Build fatbins for common Tensor Core-capable architectures instead.
# set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
include(CTest)

link_libraries(fmt::fmt)
# link_libraries(${CUDA_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(angle angle.cu)
