#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <float.h>
#include <math.h>

#include <ocllib.h>
#include <utils.h>
#include <clFFT.h>

#ifdef checkError
#undef checkError
#endif
#define checkError(status, ...) _checkFFTError(__LINE__, __FILE__, status, __VA_ARGS__)

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;

static cl_program program;
static cl_kernel kernel;
static cl_mem buffer_in_real, buffer_in_img, buffer_out_real, buffer_out_img;

void teardown(int exit_status)
{
  if (buffer_in_real) clReleaseMemObject(buffer_in_real);
  if (buffer_in_img) clReleaseMemObject(buffer_in_img);
  if (buffer_out_real) clReleaseMemObject(buffer_out_real);
  if (buffer_out_img) clReleaseMemObject(buffer_out_img);

  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  exit(exit_status);
}

void print_fft_error(cl_int status) {
  switch (status) {
    case CLFFT_BUGCHECK:
      fprintf(stderr,"CLFFT_BUGCHECK");
      break;
    case CLFFT_NOTIMPLEMENTED:
      fprintf(stderr,"CLFFT_NOTIMPLEMENTED");
      break;
    case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
      fprintf(stderr,"CLFFT_TRANSPOSED_NOTIMPLEMENTED");
      break;
    case CLFFT_FILE_NOT_FOUND:
      fprintf(stderr,"CLFFT_FILE_NOT_FOUND");
      break;
    case CLFFT_FILE_CREATE_FAILURE:
      fprintf(stderr,"CLFFT_FILE_CREATE_FAILURE");
      break;
    case CLFFT_VERSION_MISMATCH:
      fprintf(stderr,"CLFFT_VERSION_MISMATCH");
      break;
    case CLFFT_INVALID_PLAN:
      fprintf(stderr,"CLFFT_INVALID_PLAN");
      break;
    case CLFFT_DEVICE_NO_DOUBLE:
      fprintf(stderr,"CLFFT_DEVICE_NO_DOUBLE");
      break;
    case CLFFT_DEVICE_MISMATCH:
      fprintf(stderr,"CLFFT_DEVICE_MISMATCH");
      break;
    default:
      print_error(status);
  }
}

void _checkFFTError(int line, const char *file, cl_int status, const char *msg, ...) {
  if(status != CL_SUCCESS) {
    // Print line and file
    print_fft_error(status);
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

int main(int argc, char **argv) {
  cl_int status;

  const char platform_name[] = "Intel";

  if (!find_platform(platform_name, &platform)) {
    fprintf(stderr,"Error: Platform \"%s\" not found\n", platform_name);
    print_platforms();
    teardown(-1);
  }

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError (status, "Error: could not query devices");

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Error: could not create context");

  print_device_info(device, 0);

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Error: could not create command queue");

  unsigned char *data;
  unsigned char *mask;
  size_t datasize, masksize, width=512, height=512;

  if (!load_file("lena.dat", &data, &datasize)) {
    teardown(-1);
  }

  if (!load_file("mask.dat", &mask, &masksize)) {
    teardown(-1);
  }

  size_t buf_size = sizeof(cl_float) * width * height;
  size_t tmp_size = sizeof(cl_float) * width * height;

  cl_float *data_in_real  = (cl_float *) malloc(buf_size);
  cl_float *data_in_img  = (cl_float *) malloc(buf_size);
  cl_float *data_out_real = (cl_float *) malloc(buf_size);
  cl_float *data_out_img = (cl_float *) malloc(buf_size);

  if (!data_in_real || !data_in_img || !data_out_real || !data_out_img) {
    fprintf(stderr, "Error: failed to allocate data\n");
    teardown(-1);
  }

  memset(data_in_real, 0, buf_size);
  memset(data_in_img, 0, buf_size);
  memset(data_out_real, 0, buf_size);
  memset(data_out_img, 0, tmp_size);


  for (unsigned int i = 0; i < width*height; ++i) {
    data_in_real[i] = data[i];
  }

  write_bmp("lena.bmp", data_in_real, width, height, 0);

  buffer_in_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, buf_size, data_in_real, &status);
  checkError(status, "Error: could not create buffer_in");

  buffer_in_img = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, buf_size, data_in_img, &status);
  checkError(status, "Error: could not create buffer_in");

  buffer_out_real = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_out");

  buffer_out_img = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buf_size, NULL, &status);
  checkError(status, "Error: could not create buffer_out");

  clfftSetupData fft_data;
  clfftStatus fstatus;
  clfftPlanHandle plan;
  cl_event event;

  fstatus = clfftSetup(&fft_data);
  checkError(fstatus, "Error: could not setup clFFT");

  size_t size[2] = {width,height};

  fstatus = clfftCreateDefaultPlan(&plan, context, CLFFT_2D, size);
  checkError(fstatus, "Error: could not create plan");

  // 
  // setup plan
  //

  fstatus = clfftSetPlanLength(plan, CLFFT_2D, size);
  checkError(fstatus, "Error: could not set size");

  fstatus = clfftSetPlanPrecision(plan, CLFFT_SINGLE);
  checkError(fstatus, "Error: could not set precision");

  fstatus = clfftSetLayout(plan, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
  checkError(fstatus, "Error: could not set layout");

  fstatus = clfftSetResultLocation(plan, CLFFT_OUTOFPLACE);
  checkError(fstatus, "Error: could not set layout");

  size_t strides[] = {1,width};
  fstatus = clfftSetPlanInStride(plan, CLFFT_2D, strides);
  checkError(fstatus, "Error: could not set layout");

  fstatus = clfftSetPlanOutStride(plan, CLFFT_2D, strides);
  checkError(fstatus, "Error: could not set layout");

  fstatus = clfftBakePlan(plan, 1, &queue, NULL, NULL);
  checkError(fstatus, "Error: could not create plan");

  //
  // forward transformation
  //
  cl_mem buffers_in[2] = {buffer_in_real, buffer_in_img};
  cl_mem buffers_out[2] = {buffer_out_real, buffer_out_img};

  fstatus = clfftEnqueueTransform(plan,CLFFT_FORWARD, 1, &queue, 0, NULL, &event, buffers_in, buffers_out, NULL);
  checkError(fstatus, "Error: could not enqueue transformation");

  status = clEnqueueReadBuffer(queue, buffer_out_real, CL_FALSE, 0, buf_size, data_out_real, 0, NULL, NULL);
  checkError(status, "Error: could not read buffer");

  status = clEnqueueReadBuffer(queue, buffer_out_img, CL_FALSE, 0, buf_size, data_out_img, 0, NULL, NULL);
  checkError(status, "Error: could not read buffer");

  status  = clFinish(queue);
  checkError(status, "Error: could not finish queue");

  for (unsigned int i=0; i < width*height; ++i) {
    data_in_real[i] = (float) sqrt(data_out_real[i]*data_out_real[i] + data_out_img[i]*data_out_img[i]);
  }

  write_bmp("mag.bmp", data_in_real, width, height, DYNAMIC|LOG);

  //
  // apply mask
  //

  for (unsigned int i = 0; i < height; ++i) {
    for (unsigned int j = 0; j < width; ++j) {
      if (mask[i*width+j] == 0) {
        data_out_real[i*width+j] = 0.f;
        data_out_img[i*width+j] = 0.f;
      }
    }
  }
  for (unsigned int i=0; i < width*height; ++i) {
    data_in_real[i] = (float) sqrt(data_out_real[i]*data_out_real[i] + data_out_img[i]*data_out_img[i]);
  }

  write_bmp("mag_masked.bmp", data_in_real, width, height, DYNAMIC|LOG);

  //
  // write masked image
  //

  status = clEnqueueWriteBuffer(queue, buffer_out_real, CL_FALSE, 0, buf_size, data_out_real, 0, NULL, NULL);
  checkError(status, "Error: could not write masked data");

  status = clEnqueueWriteBuffer(queue, buffer_out_img, CL_FALSE, 0, buf_size, data_out_img, 0, NULL, NULL);
  checkError(status, "Error: could not write masked data");


  // 
  // reverse transformation
  //

  fstatus = clfftEnqueueTransform(plan,CLFFT_BACKWARD, 1, &queue, 0, NULL, &event, buffers_out, buffers_in, NULL);
  checkError(fstatus, "Error: could not create plan");

  //
  // read results
  //

  status = clEnqueueReadBuffer(queue, buffer_in_real, CL_TRUE, 0, buf_size, data_out_real, 0, NULL, NULL);
  checkError(status, "Error: could not read output");

  status = clEnqueueReadBuffer(queue, buffer_in_img, CL_TRUE, 0, buf_size, data_out_img, 0, NULL, NULL);
  checkError(status, "Error: could not read output");

  status  = clFinish(queue);
  checkError(status, "Error: could not finish queue");

  fstatus     = clfftDestroyPlan(&plan);
  checkError(fstatus, "Error: could not destroy plan");

  fstatus = clfftTeardown();
  checkError(fstatus, "Error: could not teardown clFFT");

  //
  // output results
  //

  write_bmp("fft.bmp", data_out_real, width, height, DYNAMIC);
  write_bmp("fft_i.bmp", data_out_img, width, height, DYNAMIC);

  free(data);
  free(mask);
  teardown(0);
}
