add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (sol-sync sync.c)
target_link_libraries (sol-sync LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
