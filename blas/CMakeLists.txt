add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (blas blas.c)
if (WIN32)
  configure_file(${PROJECT_SOURCE_DIR}/dist/${PLATFORM_PATH}/${LIB_PATH}/clBLAS.dll
    clBLAS.dll COPYONLY)
endif(WIN32)
target_link_libraries (blas LINK_PUBLIC ocllib ${OpenCL_LIBRARIES} clBLAS)
