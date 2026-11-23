cmake_minimum_required(VERSION 3.20)
project(vision_infer CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -g -O2)

add_library(vt src/conv.cpp src/pool.cpp)
target_include_directories(vt PUBLIC include)

add_executable(test_layers tests/test_layers.cpp)
target_link_libraries(test_layers PRIVATE vt)

enable_testing()
add_test(NAME layers COMMAND test_layers)
