cmake_minimum_required(VERSION 3.18)

# WMMA/Tensor Cores require SM 7.0+. Must be set before project() so nvcc uses it.
set(CMAKE_CUDA_ARCHITECTURES "86;89;90" CACHE STRING
    "CUDA architectures (WMMA requires >= 70)")

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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --ptxas-options=-v")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()
# Finds the toolkit that belongs to CMAKE_CUDA_COMPILER (the nvcc used by
# project()). Override only if that guess is wrong:
#   cmake -DCUDAToolkit_ROOT=/usr/local/cuda-12.3
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
include(CTest)

link_libraries(fmt::fmt)
# link_libraries(${CUDA_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

