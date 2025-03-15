#ifndef RAYTRACER_H
#define RAYTRACER_H
#include "obj_loader.h"

__global__ void renderKernel(uchar4 *pixels, int width, int height, Triangle *triangles, int numTriangles);

#endif