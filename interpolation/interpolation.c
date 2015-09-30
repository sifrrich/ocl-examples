#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

#include "ocllib.h"
#include <utils.h>

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;

static cl_program program;
static cl_kernel kernel;
static cl_mem buffer_in, buffer_out;

void teardown(int exit_status)
{
  if (buffer_in) clReleaseMemObject(buffer_in);
  if (buffer_out) clReleaseMemObject(buffer_out);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  exit(exit_status);
}

int main(int argc, char **argv) {
  cl_int status;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <scale>\n", argv[0]);
    teardown(-1);
  }

  cl_float scale = strtof(argv[1],NULL);
  printf("scale: %f\n", scale);

  const char *platform_name = "NVIDIA";

  if (!find_platform(platform_name, &platform)) {
    fprintf(stderr,"Error: Platform \"%s\" not found\n", platform_name);
    print_platforms();
    teardown(-1);
  }

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError (status, "Error: could not query devices");

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "could not create context");

  const char name[] = KERNELDIR "/interpolation.cl";

  unsigned char *source;
  size_t size;
  if (!load_file(name, &source, &size)) {
    teardown(-1);
  }

  program = clCreateProgramWithSource(context, 1, (const char **) &source, &size, &status);
  checkError(status, "Error: failed to create program %s: ", name);

  status = clBuildProgram(program, 1, &device, "-I.", NULL, NULL);
  if (status != CL_SUCCESS) {
    print_build_log(program, device);
    checkError(status, "Error: failed to create build %s: ", name);
  }

  free(source);

  print_device_info(device, 0);

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "could not create command queue");

  cl_ulong start, end;
  cl_event event;

  unsigned char *data;
  size_t datasize;

  if (!load_file("lena.dat", &data, &datasize)) {
    teardown(-1);
  }

  size_t width  = 512;
  size_t height = 512;

  size_t new_width = (size_t) ((int) width*scale);
  size_t new_height = (size_t) ((int) height*scale);
  printf("new size: %d %d\n", (int) new_width, (int) new_height);

  size_t buf_size = new_width*new_height*sizeof(cl_float);

  float *data_out = malloc(buf_size);
  if (!data_out) {
    fprintf(stderr,"\nError: malloc failed\n");
    teardown(-1);
  }

  kernel = clCreateKernel(program, "interpolation", &status);
  checkError(status, "could not create kernel");

  cl_image_format format = { CL_R, CL_UNORM_INT8};
  buffer_in = clCreateImage2D (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
    width, height, 0,
    data,
    &status);
  checkError(status, "Error: could not create image");

  cl_image_format format2 = { CL_R, CL_FLOAT};
  buffer_out = clCreateImage2D (context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, &format2,
    new_width, new_height, 0,
    NULL,
    &status);
  checkError(status, "Error: could not create image");

  // execute kernel
  int arg = 0;
  status  = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer_in);
  status  = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer_out);
  checkError(status, "Error: could not set args");

  size_t work_size[] = {new_width, new_height};
  size_t local_size[] = {1, 1};

  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_size, local_size, 0, NULL, &event);
  checkError(status, "Error: could not enqueue kernel");

  status = clWaitForEvents(1, &event);
  checkError(status, "Error: could not wait for event");

  status  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  checkError(status, "Error: could not get start profile information");

  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  checkError(status, "Error: could not get end profile information");

  status = clReleaseEvent(event);
  checkError(status, "Error: could not release event");

  // read results back
  size_t origin[] = {0,0,0};
  size_t region[] = {new_width, new_height, 1};
  status = clEnqueueReadImage(queue, buffer_out, CL_FALSE, origin, region, new_width*sizeof(cl_float), 0, data_out, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status  = clFinish(queue);
  checkError(status, "Error: could not finish successfully");

  double elapsed = (end - start) * 1e-9f;
  printf("time: %f\n", elapsed);

  write_bmp("scale.bmp", data_out, new_width, new_height, NORMAL);

  free(data);
  free(data_out);
  teardown(0);
}


