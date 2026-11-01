cmake_minimum_required(VERSION 3.16)
project(fast_matmul CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native")

add_library(matmul src/simple.cpp src/packed.cpp src/reference.cpp)
target_include_directories(matmul PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(matmul PUBLIC Threads::Threads)
if(APPLE)
  target_link_libraries(matmul PUBLIC "-framework Accelerate")
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(matmul PUBLIC ${BLAS_LIBRARIES})
endif()

add_executable(bench bench/bench.cpp)
target_link_libraries(bench matmul)

add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness matmul)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
