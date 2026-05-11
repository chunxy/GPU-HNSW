find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)

add_executable(bench-build-gpu bench-build-gpu.cpp)
target_link_libraries(bench-build-gpu PRIVATE Boost::program_options)

add_executable(bench-build-cpu-single bench-build-cpu-single.cpp)
target_link_libraries(bench-build-cpu-single PRIVATE Boost::program_options OpenMP::OpenMP_CXX)

add_executable(bench-build-cpu bench-build-cpu.cpp)
target_link_libraries(bench-build-cpu PRIVATE Boost::program_options OpenMP::OpenMP_CXX)

add_executable(bench-index-search bench-index-search.cpp)
target_link_libraries(bench-index-search PRIVATE Boost::program_options)
