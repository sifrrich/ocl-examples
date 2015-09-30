#define _CRT_SECURE_NO_DEPRECATE
#include <float.h>
#include <utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Taken from http://stackoverflow.com/a/2654860
 */
int  write_bmp(const char *name, float *data_in, size_t width, size_t height, int filters) {
  FILE *f;
  int filesize = 54 + 3*width*height;

  float *data = malloc(sizeof(float)*width*height);
  if (!data) {
    fprintf(stderr, "Error: malloc failed\n");
    return 0;
  }
  memcpy(data, data_in, width*height*sizeof(float));

  float max=FLT_MIN;
  float min=FLT_MAX;
  for (unsigned int i = 0; i < width*height; ++i) {
    float c = data[i];

    if (c > max) max = c;
    if (c < min) min = c;
  }

  float offset = 0.f;
  if (filters & LOG) offset = 0.01f;

  if (min <= 0) {
    for (unsigned int i = 0; i < width*height; ++i) {
      data[i] = data[i] - min + offset;
    }
    max = max - min + offset;
  }
  min=offset;

  float step;
  float delta = max-min;
  if (filters & LOG)
    step = 255.0f/((float) log(delta));
  else
    step = (float) 255.0f/(delta);

  printf("[%s] min: %f, max: %f, step: %f\n", name, min, max, step);

  int pad_size   = (4-(width*3)%4)%4;
  int row_size = width+pad_size;
  int data_size = row_size*height*3;
  unsigned char *img = (unsigned char *) malloc(data_size);
  if (!img) {
    fprintf(stderr, "Error: failed to alloc data");
    return 0;
  }
  memset(img,0,sizeof(img));

  int out_of_range=0;

  for(unsigned int i=0; i<height; i++) {
    for(unsigned int j=0; j<width; j++) {
      float color = data[i*width+j];

      if (filters & LOG) {
        color = (float) log(color);
      }

      if (filters & DYNAMIC)
        color=color * step;

      unsigned char coloru = (unsigned char) color;
      if (coloru > 255.) {
        out_of_range=1;
        coloru=255;
      }

      unsigned int x = j;
      unsigned int y = height-1-i;

      img[(y*width+x)*3+0] = coloru;
      img[(y*width+x)*3+1] = coloru;
      img[(y*width+x)*3+2] = coloru;
    }
  }
  if (out_of_range)
    fprintf(stderr, "Warning: color out of range! Better use filer DYNAMIC!\n");

  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
  unsigned char bmppad[3] = {0,0,0};

  bmpfileheader[ 2] = (unsigned char)(filesize    );
  bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
  bmpfileheader[ 4] = (unsigned char)(filesize>>16);
  bmpfileheader[ 5] = (unsigned char)(filesize>>24);

  bmpinfoheader[ 4] = (unsigned char)(width    );
  bmpinfoheader[ 5] = (unsigned char)(width>> 8);
  bmpinfoheader[ 6] = (unsigned char)(width>>16);
  bmpinfoheader[ 7] = (unsigned char)(width>>24);
  bmpinfoheader[ 8] = (unsigned char)(height    );
  bmpinfoheader[ 9] = (unsigned char)(height>> 8);
  bmpinfoheader[10] = (unsigned char)(height>>16);
  bmpinfoheader[11] = (unsigned char)(height>>24);

  f = fopen(name,"wb");
  fwrite(bmpfileheader,1,14,f);
  fwrite(bmpinfoheader,1,40,f);

  for(unsigned int i=0; i<height; ++i) {
    fwrite(img+(width*i*3),3,width,f);
    fwrite(bmppad,1, pad_size,f);
  }

  fclose(f);

  return 1;
}
