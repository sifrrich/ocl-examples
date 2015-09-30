cmake_minimum_required (VERSION 2.8)
project (OpenCL-Examples)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (CMAKE_BUILD_TYPE Release)

if (WIN32)
  set (PLATFORM_PATH win)
  set (CMAKE_FIND_LIBRARY_PREFIXES "")
  set (CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
endif (WIN32)

if (UNIX)
  set (PLATFORM_PATH lin)
  set (CMAKE_FIND_LIBRARY_PREFIXES "lib")
  set (CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
  set (CMAKE_C_FLAGS "-g -std=c99 ${CMAKE_C_FLAGS}")
endif (UNIX)

find_package(OpenCL REQUIRED)

if (NOT (${OpenCL_FOUND})
  OR ${OpenCL_INCLUDE_DIRS} STREQUAL ""
  OR (${OpenCL_LIBRARIES}) STREQUAL "")
  message (FATAL"No OpenCL found. Stop.")
endif()

include_directories (
  common
  ${PROJECT_SOURCE_DIR}/dist/${PLATFORM_PATH}/include
  ${OpenCL_INCLUDE_DIRS})

# handle 32/64-bit library includes
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set (LIB_PATH lib)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set (LIB_PATH lib64)
endif()

link_directories (${PROJECT_SOURCE_DIR}/dist/${PLATFORM_PATH}/${LIB_PATH})

macro(add_subdirectory_ifexists dir)
  if(EXISTS "${PROJECT_SOURCE_DIR}/${dir}" AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/${dir}")
    add_subdirectory (${PROJECT_SOURCE_DIR}/${dir})
  endif()
endmacro(add_subdirectory_ifexists)

# convenience functions for OpenCL kernels
add_subdirectory (common)

add_subdirectory_ifexists (matrix)
add_subdirectory_ifexists (gauss)
add_subdirectory_ifexists (sync)
add_subdirectory_ifexists (reduce)
add_subdirectory_ifexists (profile)
add_subdirectory_ifexists (interpolation)
add_subdirectory_ifexists (blas)
add_subdirectory_ifexists (fft)
