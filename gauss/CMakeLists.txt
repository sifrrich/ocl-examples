add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (gauss gauss.c)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lena.dat lena.dat COPYONLY)
target_link_libraries (gauss LINK_PUBLIC ocllib utils ${OpenCL_LIBRARIES})
