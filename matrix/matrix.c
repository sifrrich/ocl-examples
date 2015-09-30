#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define _CRT_SECURE_NO_DEPRECATE
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

#include <ocllib.h>

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;

static cl_program program;
static cl_kernel kernel;
static cl_mem buffer_A, buffer_B, buffer_C;

void teardown(int exit_status)
{
  if (buffer_A) clReleaseMemObject(buffer_A);
  if (buffer_B) clReleaseMemObject(buffer_B);
  if (buffer_C) clReleaseMemObject(buffer_C);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  exit(exit_status);
}

void matrix_mul(const float *A, const float *B, float *C, cl_int M) {
  for (int i=0; i<M; i++){
    for (int j=0; j<M; j++){
      for (int k=0; k<M; k++) {
        C[i*M+j] += A[i*M+k] * B[k*M+j];
      }
    }
  }
}

int main(int argc, char **argv) {
  cl_int status;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <kernel>\n", argv[0]);
    teardown(-1);
  }

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

  const char name[] = KERNELDIR "/matrix.cl";

  unsigned char *source;
  size_t size;
  if (!load_file(name, &source, &size)) {
    teardown(-1);
  }

  program = clCreateProgramWithSource(context, 1, (const char **) &source, &size, &status);
  checkError(status, "Error: failed to create program %s: ", name);

  status = clBuildProgram(program, 1, &device, "-I. -cl-fast-relaxed-math -cl-mad-enable -cl-nv-verbose", NULL, NULL);
  if (status != CL_SUCCESS) {
    print_build_log(program, device);
    checkError(status, "Error: failed to build program %s: ", name);
  }
  print_build_log(program, device);

  free(source);

  print_device_info(device, 0);

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "could not create command queue");

  cl_ulong start, end;
  cl_event event;

  cl_int M  = 1024;
  size_t buf_size = M*M*sizeof(cl_float);

  float *A  = malloc(buf_size);
  float *B  = malloc(buf_size);
  float *C  = malloc(buf_size);
  float *Ref  = malloc(buf_size);
  if (!A || !B || !C || !Ref) {
    fprintf(stderr,"\nError: malloc failed\n");
    teardown(-1);
  }

  memset(C, 0, buf_size);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i*M+j] = (float) i;
      B[i*M+j] = (float) j;
    }
  }

#if DEBUG
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      printf("%.2f %.2f ", A[i*M+j], B[i*M+j]);
    }
    printf("\n");
  }
#endif

  char kernelname[256];
#ifdef WIN32
  _snprintf(kernelname, 256, "matrix_mul%s", argv[1]);
#else
  snprintf(kernelname, 256, "matrix_mul%s", argv[1]);
#endif

  kernel = clCreateKernel(program, kernelname, &status);
  checkError(status, "could not create kernel %s", kernelname);

  buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_in");

  buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_out");

  buffer_C = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_out");

  status = clEnqueueWriteBuffer(queue, buffer_A, CL_FALSE, 0, buf_size, A, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, buffer_B, CL_FALSE, 0, buf_size, B, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, buffer_C, CL_FALSE, 0, buf_size, C, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  // execute kernel
  int arg = 0;
  status = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer_A);
  status = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer_B);
  status = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer_C);
  status = clSetKernelArg(kernel, arg++, sizeof(cl_int), &M);
  checkError(status, "Error: could not set args");

  size_t dim;
  size_t work_size[2];
  size_t local_size[2];

  switch(atoi(argv[1])) {
    case 1:
      dim=2;
      work_size[0]  = M;
      work_size[1]  = M;

      local_size[0] = 32;
      local_size[1] = 32;
      break;
    case 2:
      dim=2;
      work_size[0]  = M;
      work_size[1]  = M;

      local_size[0] = 32;
      local_size[1] = 32;
      break;
    case 3:
      dim=1;
      work_size[0]  = M;

      local_size[0] = 32;
      break;
    case 4:
      dim=1;
      work_size[0]  = M;

      local_size[0] = 32;
      break;
    case 5:
      dim=1;
      work_size[0]  = M;

      local_size[0] = 128;
      break;
    default:
      fprintf(stderr, "Invalid kernel number\n");
      teardown(-1);
  }

  status = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, local_size, 0, NULL, &event);
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
  status = clEnqueueReadBuffer(queue, buffer_C, CL_FALSE, 0, buf_size, C, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status  = clFinish(queue);
  checkError(status, "Error: could not finish successfully");

  double elapsed = (end - start)*1e-9;
  double gflops = M*M*M*1e-9/elapsed*2;
  printf("time: %f\n", elapsed);
  printf("gflops: %f\n", gflops);

#define CHECK
#ifdef CHECK
  matrix_mul(A,B,Ref,M);

  int correct = 1;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      if (Ref[i*M+j] != C[i*M+j]) {
        correct = 0;
      }
    }
  }

  if (!correct)
    fprintf(stderr, "Compare failed\n");
#endif


  free(A);
  free(B);
  free(C);
  free(Ref);
  teardown(0);
}


