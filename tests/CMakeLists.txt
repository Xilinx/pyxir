cmake_minimum_required(VERSION 3.5.1)
project(${TARGET}_test VERSION 0.1.0)

include_directories(${CMAKE_SOURCE_DIR}/include)
# include_directories(${CMAKE_SOURCE_DIR}/lib/catch2)

set(CMAKE_CXX_STANDARD 11)

set(TARGET pyxir)

set(TEST_DIR tests)
file(GLOB_RECURSE TEST_SOURCES "cpp/*cpp")
set(TESTS ${TEST_SOURCES})

# Generate a test executable

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/Catch2 ${CMAKE_CURRENT_BINARY_DIR}/lib/Catch2)

add_executable("${TARGET}_test" ${TEST_SOURCES})
add_dependencies("${TARGET}_test" ${TARGET})
target_link_libraries("${TARGET}_test" PUBLIC pyxir Catch2::Catch2)

set_target_properties("${TARGET}_test" PROPERTIES VERSION ${PROJECT_VERSION})