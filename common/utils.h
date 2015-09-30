#ifndef UTILS_H
#define UTILS_H

#ifdef _WIN32
#include <BaseTsd.h>
#else
#include <stddef.h>
#endif

enum {
  NORMAL=0,
  LOG=1,
  DYNAMIC=2, // Map colors 0 ... 255
} FILTERS;

int write_bmp(const char *name, float *data, size_t width, size_t height, int filters);

#endif /* UTILS_H */
