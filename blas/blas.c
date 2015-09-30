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

#include <ocllib.h>
#include <clBLAS.h>

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

  const char platform_name[] = "NVIDIA";

  if (!find_platform(platform_name, &platform)) {
    fprintf(stderr,"Error: Platform \"%s\" not found\n", platform_name);
    print_platforms();
    teardown(-1);
  }

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError (status, "Error: could not query devices");

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "could not create context");


  print_device_info(device, 0);

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "could not create command queue");

  cl_ulong start, end;
  cl_event event;

  cl_int M = 512;
  size_t buf_size = M*M*sizeof(cl_float);

  float *A = malloc(buf_size);
  float *B = malloc(buf_size);
  float *C = malloc(buf_size);
  float *Ref  = malloc(buf_size);
  if (!A || !B || !C || !Ref) {
    fprintf(stderr,"\nError: malloc failed\n");
    teardown(-1);
  }

  memset(C, 0, buf_size);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i*M+j] = (float) i / 1024.;
      B[i*M+j] = (float) j / 1024.;
    }
  }

  buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_in");

  buffer_B = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_in");

  buffer_C = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_in");

  //
  // Init Matrices
  //

  status = clEnqueueWriteBuffer(queue, buffer_A, CL_FALSE, 0, buf_size, A, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  status = clEnqueueWriteBuffer(queue, buffer_B, CL_FALSE, 0, buf_size, B, 0, NULL, NULL);
  checkError(status, "Error: could not copy data into device");

  //
  // Setup BLAS
  //

  status = clblasSetup();
  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: clblasSetup() failed with %d\n", status);
    teardown(-1);
  }

  status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, M, M,
                         1.0, buffer_A, 0, M,
                         buffer_B, 0, M, 1.0,
                         buffer_C, 0, M,
                         1, &queue, 0, NULL, &event);

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Error: clblasSgemmEx() failed with %d\n", status);
    teardown(-1);
  }

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

  double elapsed = (end - start) * 1e-9f;
  double gflops = M*M*M*1e-9/elapsed*2;
  printf("time: %f\n", elapsed);
  printf("gflops: %f\n", gflops);

#define CHECK
#ifdef CHECK
  matrix_mul(A,B,Ref,M);

  int correct = 1;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      if (abs(Ref[i*M+j]) - abs(C[i*M+j]) > 0.005) {
        correct = 0;
        printf("at %d,%d: %f != %f\n", i, j, Ref[i*M+j], C[i*M+j]);
        goto endloop;
      }
    }
  }
endloop:

  if (!correct)
    fprintf(stderr, "Compare failed\n");
#endif

  free(A);
  free(B);
  free(C);
  free(Ref);
  teardown(0);
}


