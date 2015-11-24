add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (sync sync.c)
target_link_libraries (sync LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
