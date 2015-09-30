#ifndef OCLLIB_H
#define OCLLIB_H

#include <stdarg.h>

#include <CL/cl.h>

#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)

void _checkError(int line, const char *file, cl_int error, const char *msg, ...);

void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
void device_info_string( cl_device_id device, cl_device_info param, const char* name);

void print_error(cl_int error);
void print_device_info( cl_device_id device, int printShort );

void print_platforms(void);
int find_platform(const char *name, cl_platform_id *platform);
cl_platform_id select_platform(const unsigned int index);

int load_file(const char *name, unsigned char **binary, size_t *size);

void print_build_log(cl_program, cl_device_id);

void teardown(int);

#endif /* OCLLIB_H */
