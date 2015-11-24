add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (interpolation interpolation.c)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lena.dat lena.dat COPYONLY)
target_link_libraries (interpolation LINK_PUBLIC ocllib utils ${OpenCL_LIBRARIES})
