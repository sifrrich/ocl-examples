add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (sol-reduce reduce.c)
target_link_libraries (sol-reduce LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
