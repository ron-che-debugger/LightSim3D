#ifndef RAYTRACER_H
#define RAYTRACER_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "math_utils.h"
#include "obj_loader.h"
#include "ray.h"

__global__ void  renderKernel(uchar4* pixels, int width, int height, Triangle* triangles, int numTriangles, float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch);

#endif