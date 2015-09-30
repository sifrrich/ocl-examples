add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (matrix matrix.c)
target_link_libraries (matrix LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
