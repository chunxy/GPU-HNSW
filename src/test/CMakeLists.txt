add_executable(test-kernels test-kernels.cu)
target_link_libraries(test-kernels PRIVATE CUDA::cudart IMPL)
add_test(NAME test-kernels COMMAND test-kernels)
set_tests_properties(test-kernels PROPERTIES SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
