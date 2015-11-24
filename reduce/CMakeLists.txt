add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (reduce reduce.c)
target_link_libraries (reduce LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
