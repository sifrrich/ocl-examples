add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (fft fft.c)
if (WIN32)
  configure_file(${PROJECT_SOURCE_DIR}/dist/${PLATFORM_PATH}/${LIB_PATH}/clFFT.dll
    clFFT.dll COPYONLY)
endif(WIN32)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mask.dat mask.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lena.dat lena.dat COPYONLY)
target_link_libraries (fft LINK_PUBLIC ocllib utils ${OpenCL_LIBRARIES} clFFT)
