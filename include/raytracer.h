#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "bvh.h"
#include "math_utils.h"
#include "obj_loader.h"
#include "pbr_utils.h"
#include "ray.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * @brief Test whether a ray intersects with a triangle.
 *
 * @param ray The ray to test.
 * @param tri The triangle to test against.
 * @param t Output parameter that stores the hit distance along the ray if an intersection occurs.
 * @return True if the ray intersects the triangle, false otherwise.
 */
__device__ bool intersectTriangle(const Ray &ray, const Triangle &tri, float &t);

/**
 * @brief Test whether a ray intersects with an axis-aligned bounding box.
 *
 * @param ray The ray to test.
 * @param box The AABB to test against.
 * @param t_min Minimum valid ray distance.
 * @param t_max Maximum valid ray distance.
 * @return True if the ray intersects the AABB within the range, false otherwise.
 */
__device__ bool intersectAABB(const Ray &ray, AABB box, float t_min, float t_max);

/**
 * @brief Initialize the random number generator states for each pixel.
 *
 * @param randStates Pointer to the array of curandState objects.
 * @param width Width of the output image.
 * @param height Height of the output image.
 */
__global__ void initRandomStates(curandState *randStates, int width, int height);

/**
 * @brief Main CUDA kernel that performs path tracing for each pixel.
 *
 * @param pixels Output buffer for the final image in uchar4 format.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param bvhNodes Pointer to the BVH node array.
 * @param triangleIndices Index mapping from BVH leaves to triangles.
 * @param triangles Scene geometry.
 * @param rootIndex Index of the root node in the BVH.
 * @param randStates Random state buffer for pixel sampling.
 * @param cameraPos Camera position in world space.
 * @param cameraDir Forward viewing direction of the camera.
 * @param objectYaw Yaw angle for camera or scene rotation.
 * @param objectPitch Pitch angle for camera or scene rotation.
 * @param lightTriangles Array of emissive triangles in the scene.
 * @param numLights Number of emissive triangles.
 */
__global__ void renderKernel(uchar4 *pixels, int width, int height,
                             BVHNode *bvhNodes, int *triangleIndices, Triangle *triangles, int rootIndex,
                             curandState *randStates,
                             float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch,
                             Triangle *lightTriangles, int numLights);

#endif