file(GLOB_RECURSE native_srcs *.cc *.cpp *.cu)
add_library(IMPL STATIC ${native_srcs})
target_include_directories(IMPL PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(IMPL PUBLIC CUDA::cudart)