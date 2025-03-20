#ifndef RAYTRACER_H
#define RAYTRACER_H
#include "bvh.h"
#include "math_utils.h"
#include "obj_loader.h"
#include "ray.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initRandomStates(curandState *randStates, int width, int height);

__global__ void renderKernel(uchar4 *pixels, int width, int height,
                             BVHNode *bvhNodes, int *triangleIndices, Triangle *triangles, int rootIndex,
                             curandState *randStates,
                             float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch);
#endif