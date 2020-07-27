cmake_minimum_required(VERSION 3.5.1)

set(TARGET pyxir)
project(${TARGET}_test)

set(CMAKE_CXX_STANDARD 11)
set(PYXIR_SHARED_LIB libpyxir.so)

find_package(PythonInterp 3.6 REQUIRED)

if(PYTHONINTERP_FOUND)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pyxir as px; print(px.get_include_dir()); print(px.get_lib_dir());"
    RESULT_VARIABLE __result
    OUTPUT_VARIABLE __output
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(__result MATCHES 0)
    string(REGEX REPLACE ";" "\\\\;" __values ${__output})
    string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
    list(GET __values 0 PYXIR_INCLUDE_DIR)
    list(GET __values 1 PYXIR_LIB_DIR)
  endif()
else()
  message(STATUS "To find Pyxir, Python interpretater is required to be found.")
endif()

add_library(pyxir SHARED IMPORTED)
message("-- Found Pyxir lib: ${PYXIR_LIB_DIR}/${PYXIR_SHARED_LIB}")
set_property(TARGET pyxir PROPERTY IMPORTED_LOCATION ${PYXIR_LIB_DIR}/${PYXIR_SHARED_LIB})

message("-- Found Pyxir include: ${PYXIR_INCLUDE_DIR}")
include_directories(${PYXIR_INCLUDE_DIR})


set(TEST_DIR tests)
file(GLOB_RECURSE TEST_SOURCES "cpp/*cpp")
set(TESTS ${TEST_SOURCES})

# Generate a test executable

add_subdirectory(${CMAKE_SOURCE_DIR}/../lib/Catch2 ${CMAKE_CURRENT_BINARY_DIR}/lib/Catch2)

add_executable("${TARGET}_test" ${TEST_SOURCES})
add_dependencies("${TARGET}_test" ${TARGET})
target_link_libraries("${TARGET}_test" PUBLIC pyxir Catch2::Catch2)

set_target_properties("${TARGET}_test" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/cpp)
