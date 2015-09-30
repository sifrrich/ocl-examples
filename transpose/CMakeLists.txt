add_definitions (-DKERNELDIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (comp comp.c)
target_link_libraries (comp LINK_PUBLIC ocllib ${OpenCL_LIBRARIES})
