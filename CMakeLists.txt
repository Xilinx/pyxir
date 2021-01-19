cmake_minimum_required(VERSION 3.5.1)
project(pyxir)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(TARGET pyxir)
set(pyxir_EXT_LIBRARIES "")
set(pyxir_EXT_DEPENDENCIES "")

option(USE_VAI_RT "Build with Vitis-AI Runtime" OFF)

find_package(PythonInterp 3.6 REQUIRED)

add_subdirectory(lib/pybind11)
include_directories(include)

add_definitions(-DTEST_FEATURE=1)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

if (DEBUG)
  add_definitions(-DDEBUG=1)
endif()

if (USE_VAI_RT)
  set(vitisai_ROOT_DIR $ENV{CONDA_PREFIX})
  set(vitisai_INCLUDE_DIRS ${vitisai_ROOT_DIR}/lib/python3.6/site-packages/vai/dpuv1/rt/vitis/include/dpu)
  set(vitisai_LIBRARIES ${vitisai_ROOT_DIR}/lib)
  set(VITISAI_SHARED_LIB libxfdnn.so)  
  include_directories(${vitisai_INCLUDE_DIRS})
  add_library(vitisai_rt SHARED IMPORTED)
  set_property(TARGET vitisai_rt PROPERTY IMPORTED_LOCATION ${vitisai_LIBRARIES}/${VITISAI_SHARED_LIB})
  list(APPEND pyxir_EXT_LIBRARIES vitisai_rt)
  # list(APPEND pyxir_EXT_DEPENDENCIES vitisai_rt)
  add_definitions(-DUSE_VAI_RT=1)

  # Remove VAI_DEFAULT backend source files
  file(GLOB_RECURSE vitisai_default_SOURCES "src/pyxir/runtime/backends/vai_default/*.cpp")
  list(REMOVE_ITEM SOURCES ${vitisai_default_SOURCES})

elseif (USE_VAI_RT_DPUCZDX8G)
  set(vitisai_INCLUDE_DIRS /usr/include/vai)
  set(vitisai_LIBRARIES /usr/lib)
  set(VITISAI_SHARED_LIB libn2cube.so)
  include_directories(${vitisai_INCLUDE_DIRS})
  add_library(vitisai_rt SHARED IMPORTED)
  set_property(TARGET vitisai_rt PROPERTY IMPORTED_LOCATION ${vitisai_LIBRARIES}/${VITISAI_SHARED_LIB})
  list(APPEND pyxir_EXT_LIBRARIES vitisai_rt)
  add_definitions(-DUSE_VAI_RT_DPUCZDX8G=1)

  # Remove VAI_DEFAULT backend source files
  file(GLOB_RECURSE vitisai_default_SOURCES "src/pyxir/runtime/backends/vai_default/*.cpp")
  list(REMOVE_ITEM SOURCES ${vitisai_default_SOURCES})

elseif (USE_VAI_RT_DPUCAHX8H)

  set(vitisai_ROOT_DIR $ENV{CONDA_PREFIX})
  set(vitisai_INCLUDE_DIRS ${vitisai_ROOT_DIR}/lib/python3.6/site-packages/vai/dpuv1/rt/vitis/include/dpu)
  #set(vitisai_LIBRARIES ${vitisai_ROOT_DIR}/lib)
  set(vitisai_LIBRARIES /usr/lib)
  set(VITISAI_SHARED_LIB libvart-runner.so)  
  include_directories(${vitisai_INCLUDE_DIRS})
  add_library(vitisai_rt SHARED IMPORTED)
  set_property(TARGET vitisai_rt PROPERTY IMPORTED_LOCATION ${vitisai_LIBRARIES}/${VITISAI_SHARED_LIB})
  list(APPEND pyxir_EXT_LIBRARIES vitisai_rt)
  # list(APPEND pyxir_EXT_DEPENDENCIES vitisai_rt)
  add_definitions(-DUSE_VAI_RT_DPUCAHX8H=1)

  # Remove VAI_DEFAULT backend source files
  file(GLOB_RECURSE vitisai_default_SOURCES "src/pyxir/runtime/backends/vai_default/*.cpp")
  list(REMOVE_ITEM SOURCES ${vitisai_default_SOURCES})
  file(GLOB_RECURSE vitisai_default_SOURCES "src/pyxir/runtime/backends/vai_rt/*.cpp")
  list(REMOVE_ITEM SOURCES ${vitisai_default_SOURCES})

else()
  # Remove VAI_RT backend source files
  file(GLOB_RECURSE vitisai_rt_SOURCES "src/pyxir/runtime/backends/vai_rt/*.cpp")
  list(REMOVE_ITEM SOURCES ${vitisai_rt_SOURCES})
endif()

add_library(${TARGET} SHARED ${SOURCES})

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(${TARGET} PUBLIC OpenMP::OpenMP_CXX)
# endif()

# add_dependencies(${TARGET} ${pyxir_EXT_DEPENDENCIES})
target_link_libraries(${TARGET} PUBLIC ${pyxir_EXT_LIBRARIES} PRIVATE pybind11::embed ${CMAKE_DL_LIBS})
# set_target_properties(${TARGET} PROPERTIES VERSION ${PROJECT_VERSION})

install(TARGETS ${TARGET} LIBRARY DESTINATION ".")
install(
    DIRECTORY "include" # source directory
    DESTINATION "." # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
)
