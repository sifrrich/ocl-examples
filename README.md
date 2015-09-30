# ocl-examples
This repository includes several examples using OpenCL.

## Build
``cmake`` is used to generate either ``Makefiles`` or projects for Visual
Studio.

Out-of-source build is recommended, e.g.
```Shell
mkdir ocl-examples-build
cd ocl-examples-build
cmake ocl-examples/CMakeLists.txt
```
For fft and blas, ``clFFT`` and ``clBLAS`` have to be available for linking.
If the libraries are not installed system-wide, they have to be placed in the
``dist``-directory.  
The file ``dist/tree`` shows the directory structure expected.

## Example Image Format
Some examples contain ``.dat`` images. These are essentially `pgm` images with
stripped headers, containing only raw pixels, one byte per pixel, in the range of
``0...ff`` to simplify reading images.

Results are generated using Bitmap file format.

## Examples
- **transpose:**  
  Simple Matrix transposition using only global memory.

- **matrix:**  
  Different implementations of matrix-matrix multiplication.
  The examples are inspired by
  <http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf>.

- **sync:**  
  Reduction in shared memory to demonstrate ``barrier`` functions to synchronize
  work items inside a work group.

- **reduce:**  
  Parallel reduction, inspired by
  <http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/>.

- **gauss:**  
  Demonstrate data transfer between OpenCL images and normal buffers using a
  gauss filter to smooth a distorded image.

- **interpolation:**  
  Enlarge/reduce the size of an image using OpenCL images.

- **blas:**  
  Matrix-Matrix multiplication using clBLAS.

- **fft:**  
  Remove a regular distortion pattern from an image using forward- and backward
  FFT-transformation with the external library clFFT.

