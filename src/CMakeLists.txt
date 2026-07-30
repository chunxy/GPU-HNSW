add_subdirectory(impl)
link_libraries(IMPL)
add_subdirectory(benchs)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

add_executable(compute-groundtruth compute-groundtruth.cpp)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
target_link_libraries(compute-groundtruth PRIVATE Boost::program_options OpenMP::OpenMP_CXX)
