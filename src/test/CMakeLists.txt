add_executable(test-kernels test-kernels.cu)
target_link_libraries(test-kernels PRIVATE CUDA::cudart IMPL)
add_test(NAME test-kernels COMMAND test-kernels)
set_tests_properties(test-kernels PROPERTIES SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")

add_executable(test-ip-vs-mm test-ip-vs-mm.cu)
target_link_libraries(test-ip-vs-mm PRIVATE CUDA::cudart)
add_test(NAME test-ip-vs-mm COMMAND test-ip-vs-mm --dim 256 --pairs 64 --repeats 1 --warmup 0)
set_tests_properties(test-ip-vs-mm PROPERTIES SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
