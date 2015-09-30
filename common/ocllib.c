#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <io.h>
#define alloca _alloca
#define access _access
#define R_OK 4
#else
#include <unistd.h>
#include <alloca.h>
#endif

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <CL/cl.h>

#include <ocllib.h>

void _checkError(int line, const char *file, cl_int status, const char *msg, ...) {
  if(status != CL_SUCCESS) {
    // Print line and file
    print_error(status);
    fprintf(stderr,"\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vfprintf(stderr, msg, vl);
    fprintf(stderr, "\n");
    va_end(vl);

    teardown(-1);
  }
}

void print_error(cl_int error) {
  switch(error) {
    case -1:
      fprintf(stderr,"CL_DEVICE_NOT_FOUND ");
      break;
    case -2:
      fprintf(stderr,"CL_DEVICE_NOT_AVAILABLE ");
      break;
    case -3:
      fprintf(stderr,"CL_COMPILER_NOT_AVAILABLE ");
      break;
    case -4:
      fprintf(stderr,"CL_MEM_OBJECT_ALLOCATION_FAILURE ");
      break;
    case -5:
      fprintf(stderr,"CL_OUT_OF_RESOURCES ");
      break;
    case -6:
      fprintf(stderr,"CL_OUT_OF_HOST_MEMORY ");
      break;
    case -7:
      fprintf(stderr,"CL_PROFILING_INFO_NOT_AVAILABLE ");
      break;
    case -8:
      fprintf(stderr,"CL_MEM_COPY_OVERLAP ");
      break;
    case -9:
      fprintf(stderr,"CL_IMAGE_FORMAT_MISMATCH ");
      break;
    case -10:
      fprintf(stderr,"CL_IMAGE_FORMAT_NOT_SUPPORTED ");
      break;
    case -11:
      fprintf(stderr,"CL_BUILD_PROGRAM_FAILURE ");
      break;
    case -12:
      fprintf(stderr,"CL_MAP_FAILURE ");
      break;
    case -13:
      fprintf(stderr,"CL_MISALIGNED_SUB_BUFFER_OFFSET ");
      break;
    case -14:
      fprintf(stderr,"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
      break;
    case -30:
      fprintf(stderr,"CL_INVALID_VALUE ");
      break;
    case -31:
      fprintf(stderr,"CL_INVALID_DEVICE_TYPE ");
      break;
    case -32:
      fprintf(stderr,"CL_INVALID_PLATFORM ");
      break;
    case -33:
      fprintf(stderr,"CL_INVALID_DEVICE ");
      break;
    case -34:
      fprintf(stderr,"CL_INVALID_CONTEXT ");
      break;
    case -35:
      fprintf(stderr,"CL_INVALID_QUEUE_PROPERTIES ");
      break;
    case -36:
      fprintf(stderr,"CL_INVALID_COMMAND_QUEUE ");
      break;
    case -37:
      fprintf(stderr,"CL_INVALID_HOST_PTR ");
      break;
    case -38:
      fprintf(stderr,"CL_INVALID_MEM_OBJECT ");
      break;
    case -39:
      fprintf(stderr,"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
      break;
    case -40:
      fprintf(stderr,"CL_INVALID_IMAGE_SIZE ");
      break;
    case -41:
      fprintf(stderr,"CL_INVALID_SAMPLER ");
      break;
    case -42:
      fprintf(stderr,"CL_INVALID_BINARY ");
      break;
    case -43:
      fprintf(stderr,"CL_INVALID_BUILD_OPTIONS ");
      break;
    case -44:
      fprintf(stderr,"CL_INVALID_PROGRAM ");
      break;
    case -45:
      fprintf(stderr,"CL_INVALID_PROGRAM_EXECUTABLE ");
      break;
    case -46:
      fprintf(stderr,"CL_INVALID_KERNEL_NAME ");
      break;
    case -47:
      fprintf(stderr,"CL_INVALID_KERNEL_DEFINITION ");
      break;
    case -48:
      fprintf(stderr,"CL_INVALID_KERNEL ");
      break;
    case -49:
      fprintf(stderr,"CL_INVALID_ARG_INDEX ");
      break;
    case -50:
      fprintf(stderr,"CL_INVALID_ARG_VALUE ");
      break;
    case -51:
      fprintf(stderr,"CL_INVALID_ARG_SIZE ");
      break;
    case -52:
      fprintf(stderr,"CL_INVALID_KERNEL_ARGS ");
      break;
    case -53:
      fprintf(stderr,"CL_INVALID_WORK_DIMENSION ");
      break;
    case -54:
      fprintf(stderr,"CL_INVALID_WORK_GROUP_SIZE ");
      break;
    case -55:
      fprintf(stderr,"CL_INVALID_WORK_ITEM_SIZE ");
      break;
    case -56:
      fprintf(stderr,"CL_INVALID_GLOBAL_OFFSET ");
      break;
    case -57:
      fprintf(stderr,"CL_INVALID_EVENT_WAIT_LIST ");
      break;
    case -58:
      fprintf(stderr,"CL_INVALID_EVENT ");
      break;
    case -59:
      fprintf(stderr,"CL_INVALID_OPERATION ");
      break;
    case -60:
      fprintf(stderr,"CL_INVALID_GL_OBJECT ");
      break;
    case -61:
      fprintf(stderr,"CL_INVALID_BUFFER_SIZE ");
      break;
    case -62:
      fprintf(stderr,"CL_INVALID_MIP_LEVEL ");
      break;
    case -63:
      fprintf(stderr,"CL_INVALID_GLOBAL_WORK_SIZE ");
      break;
    default:
      fprintf(stderr,"UNRECOGNIZED ERROR CODE (%d)", error);
  }
}

void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
  cl_ulong a;
  clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
  fprintf(stderr,"%-40s = %lu\n", name, a);
}

void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
  cl_uint a;
  clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
  fprintf(stderr,"%-40s = %u\n", name, a);
}

void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
  cl_bool a;
  clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
  fprintf(stderr,"%-40s = %s\n", name, (a?"true":"false"));
}

void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
  size_t sz;
  clGetDeviceInfo(device, param, 0, NULL, &sz);

  char *a = (char *) alloca(sz);
  clGetDeviceInfo(device, param, sz, a, NULL);
  fprintf(stderr,"%-40s = %s\n", name, a);
}

void platform_info_string( cl_platform_id platform, cl_platform_info param, const char* name) {
  size_t sz;
  clGetPlatformInfo(platform, param, 0, NULL, &sz);

  char *a = (char *) alloca(sz);
  clGetPlatformInfo(platform, param, sz, a, NULL);
  fprintf(stderr,"%-40s = %s\n", name, a);
}

void print_platform_info(cl_platform_id platform) {
  fprintf(stderr,"Platform Info:\n");
  fprintf(stderr,"===============================================\n");

  platform_info_string(platform, CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
  platform_info_string(platform, CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
  platform_info_string(platform, CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
  platform_info_string(platform, CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
  fprintf(stderr,"===============================================\n");
}

void print_device_info(cl_device_id device, int printShort) {
  fprintf(stderr,"Device Info:\n");
  fprintf(stderr,"===============================================\n");
  device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
  device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
  if (printShort) {
    fprintf(stderr,"===============================================\n");
    return;
  }

  device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

  cl_command_queue_properties ccp;
  clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
  fprintf(stderr,"%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
  fprintf(stderr,"%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
  fprintf(stderr,"===============================================\n");
}


int find_platform(const char *platform_name_search, cl_platform_id *platform) {
  cl_int status;

  // Get number of platforms.
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: query for number of platforms failed\n");
    return 0;
  }

  // Get a list of all platform ids.
  cl_platform_id *pids = (cl_platform_id *) alloca(sizeof(cl_platform_id)*num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: query for all platform ids failed");
    return 0;
  }

  // For each platform, get name and compare against the search string.
  for(unsigned i = 0; i < num_platforms; ++i) {
    size_t sz;
    status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, 0, NULL, &sz);
    if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: query for size of platform name failed");
      continue;
    }

    char *name = (char *) alloca(sz);
    status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, sz, name, NULL);
    if (status != CL_SUCCESS) {
      fprintf(stderr, "Error: query for platform name failed");
      continue;
    }

    if ( strstr(name, platform_name_search) ) {
      *platform = pids[i];
      return 1;
    }
  }
  return 0;
}

cl_platform_id select_platform(const unsigned int id) {
  cl_int status;

  // Get number of platforms.
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  checkError(status, "Query for number of platforms failed");

  // Get a list of all platform ids.
  cl_platform_id *pids = (cl_platform_id *) alloca(sizeof(cl_platform_id)*num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);
  checkError(status, "Query for all platform ids failed");

  if (id < num_platforms)
    return pids[id];

  return NULL;
}

void print_platforms(void) {

  // Get number of platforms.
  cl_uint num_platforms;
  cl_int status;

  status = clGetPlatformIDs(0, NULL, &num_platforms);
  checkError(status, "Query for number of platforms failed");

  fprintf(stderr,"%d Platforms found\n", num_platforms);

  // Get a list of all platform ids.
  cl_platform_id *pids = (cl_platform_id *) alloca(sizeof(cl_platform_id)*num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);
  checkError(status, "Query for all platform ids failed");

  // For each platform, get name.
  for(unsigned i = 0; i < num_platforms; ++i) {
    fprintf(stderr,"Platform ID %d\n", i);

    print_platform_info(pids[i]);

    //get all device ids
    cl_uint num_devices;
    status = clGetDeviceIDs(pids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    checkError(status, "Query for number of devices failed");

    cl_device_id *dids = (cl_device_id *) alloca(sizeof(cl_device_id)*num_devices);

    status = clGetDeviceIDs(pids[i], CL_DEVICE_TYPE_ALL, num_devices, dids, NULL);
    checkError(status, "Query for device ids");

    for (unsigned j = 0; j < num_devices; ++j) {
      device_info_string(dids[j], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
      device_info_string(dids[j], CL_DEVICE_NAME, "CL_DEVICE_NAME");
      device_info_string(dids[j], CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
      device_info_string(dids[j], CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
    }
    fprintf(stderr,"===============================================\n");
  }
}

int load_file(const char *name, unsigned char **binary, size_t *size) {
  FILE *fp;
  unsigned char *ptr;
  size_t s;

  if (access(name, R_OK)) {
    fprintf(stderr, "Error: failed to access %s\n", name);
    goto error_ret;
  }

  // Load the binary.
  fp = fopen(name, "rb");
  if(!fp) {
    fprintf(stderr, "Error: fopen %s failed: %s\n", name, strerror(errno));
    goto error_ret;
  }

  // Get the size of the file
  if (fseek(fp, 0, SEEK_END)) {
    fprintf(stderr, "Error: fseek %s failed: %s\n", name, strerror(errno));
    goto error_close;
  }
  s = ftell(fp);

  // Allocate space for the binary
  ptr = (unsigned char *) malloc(s);
  if (!ptr) {
    fprintf(stderr, "Error: malloc %s failed: %s\n", name, strerror(errno));
    goto error_ret;
  }

  // Go back to the file start
  rewind(fp);

  // Read the file into the binary
  if(fread((void*)ptr, 1, s, fp) != s) {
    fprintf(stderr, "Error: fread %s failed: %s\n", name, strerror(errno));
    goto error_free;
  }

  *size = s;
  *binary = ptr;

  return 1;

error_free:
  free(ptr);
error_close:
  fclose(fp);
error_ret:
  *binary = NULL;
  *size   = 0;
  return 0;
}

int create_program(const char *name, cl_program *program, cl_context context,
    cl_device_id device, const char *compiler_opts){
  cl_int status;

  unsigned char *binary;
  size_t size;
  if (!load_file(name, &binary, &size)) goto error_ret;
  if (!binary) goto error_ret;

  *program = clCreateProgramWithSource(context, 1, (const char **) &binary, &size, &status);

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: failed to create program %s:", name);
    print_error(status);
    goto error_free;
  }

  status = clBuildProgram(*program, 1, &device, compiler_opts, NULL, NULL);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: failed to build program %s\n", name);
    goto error_free;
  }

  cl_build_status build_status;

  clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_STATUS,
      sizeof(cl_build_status), &build_status, NULL);

  if(build_status != CL_SUCCESS) {
    goto error_free;
  }

  free(binary);
  return 1;

error_free:
  free(binary);
error_ret:
  return 0;
}

void print_build_log(cl_program program, cl_device_id device) {
  size_t log_size;

  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  char *log = (char*) malloc(log_size);
  if (!log) {
    fprintf(stderr, "Error: Could not allocate memory for OpenCL-Compiler Log.");
  }

  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

  fprintf(stdout,"%s\n", log);

  return;
}
