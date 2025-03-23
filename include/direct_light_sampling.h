#ifndef DIRECT_LIGHT_SAMPLING_H
#define DIRECT_LIGHT_SAMPLING_H

/**
 * @file direct_light_sampling.h
 * @brief Implements direct lighting estimation using triangle lights and BVH-based visibility checks.
 *
 * This header provides device-side functions for:
 * - Shadow ray occlusion testing using BVH traversal.
 * - Monte Carlo sampling of triangle lights for estimating direct illumination.
 *
 * Used in GPU ray tracers to compute realistic lighting from emissive surfaces.
 */
#include "bvh.h"
#include "math_utils.h"
#include "obj_loader.h"
#include "pbr_utils.h"
#include "ray.h"
#include "raytracer.h"
#include <curand_kernel.h>

/**
 * @brief Check if a shadow ray is occluded by any geometry before reaching its target.
 *
 * A ray is considered occluded if it intersects any triangle before reaching `maxT`.
 * This is used to determine shadowing in direct lighting.
 *
 * @param ray: The shadow ray to test.
 * @param maxT: The maximum distance the ray can travel before hitting the light.
 * @param bvhNodes: The array of BVH nodes for spatial hierarchy traversal.
 * @param triangleIndices: The index mapping to triangles within BVH leaves.
 * @param triangles: The triangle geometry list.
 * @param rootIndex: The root node index of the BVH.
 * @return true if the ray is occluded; false if the path to the light is clear.
 */
__device__ inline bool isOccluded(const Ray &ray, float maxT,
                                  BVHNode *bvhNodes, int *triangleIndices, Triangle *triangles, int rootIndex) {

    const int stackSize = 64;
    int stack[stackSize];
    int stackPtr = 0;
    stack[stackPtr++] = rootIndex;
    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        BVHNode node = bvhNodes[nodeIndex];
        if (!intersectAABB(ray, node.bbox, 0.001f, maxT))
            continue;
        if (node.isLeaf) {
            for (int i = node.start; i < node.start + node.count; i++) {
                int triIndex = triangleIndices[i];
                float t;
                if (intersectTriangle(ray, triangles[triIndex], t) && t < maxT)
                    return true;
            }
        } else {
            if (node.left != -1)
                stack[stackPtr++] = node.left;
            if (node.right != -1)
                stack[stackPtr++] = node.right;
        }
    }
    return false;
}

/**
 * @brief Estimate direct lighting at a surface point using one-sample Monte Carlo integration.
 *
 * It gives a single-sample estimate of how much direct lighting a point receives from the environment's triangle lights.
 * The probability density is assumed to be uniform over the list of lights and uniformly on the triangle surface.
 *
 * @param hitPoint: The point on the surface being shaded.
 * @param normal: The surface normal at the hit point.
 * @param bvhNodes: The array of BVH nodes for scene geometry.
 * @param triangleIndices: The triangle index mapping used by BVH leaves.
 * @param triangles: All triangle geometry in the scene.
 * @param rootIndex: Root node index of the BVH.
 * @param lightTriangles: List of triangle lights in the scene.
 * @param numLights: Total number of triangle lights.
 * @param randState: Random number generator state for sampling.
 * @return The estimated incoming radiance from direct lighting as a float3 RGB color.
 */
__device__ inline float3 sampleDirectLight(
    float3 hitPoint, float3 normal,
    BVHNode *bvhNodes, int *triangleIndices, Triangle *triangles, int rootIndex,
    Triangle *lightTriangles, int numLights,
    curandState *randState) {

    if (numLights == 0)
        return make_float3(0, 0, 0);

    // Randomly select one light triangle
    int lightIndex = min((int)(curand_uniform(randState) * numLights), numLights - 1);
    Triangle light = lightTriangles[lightIndex];

    // Uniformly sample a point on the triangle using barycentrics
    float r1 = curand_uniform(randState);
    float r2 = curand_uniform(randState);
    float sqrt_r1 = sqrtf(r1);
    float u = 1.0f - sqrt_r1;        // First barycentric weight
    float v = sqrt_r1 * (1.0f - r2); // Second barycentric weight
    float w = sqrt_r1 * r2;          // Third barycentric weight
    float3 lightPos = MathUtils::float3_add(
        MathUtils::float3_add(
            MathUtils::float3_scale(light.v0, u),
            MathUtils::float3_scale(light.v1, v)),
        MathUtils::float3_scale(light.v2, w));

    // Compute direction and distance to light sample
    float3 lightDir = MathUtils::float3_subtract(lightPos, hitPoint);
    float distance = sqrtf(MathUtils::dot(lightDir, lightDir));
    lightDir = MathUtils::normalize(lightDir);

    // Compute the cosine of the angle between the surface normal and the light direction.
    // This tells us if the surface is facing the light (cosTheta > 0).
    float cosTheta = fmaxf(MathUtils::dot(normal, lightDir), 0.0f);

    // Compute the cosine of the angle between the light's normal and the direction to the surface.
    // This checks whether the front face of the light is facing the surface (cosThetaLight > 0).
    float cosThetaLight = fmaxf(MathUtils::dot(light.normal, MathUtils::float3_scale(lightDir, -1.0f)), 0.0f);

    // If either the surface is facing away from the light, or
    // the light is facing away from the surface (i.e., it's the back face), discard this sample.
    if (cosTheta <= 0.0f || cosThetaLight <= 0.0f)
        return make_float3(0, 0, 0);

    // Compute area of the light triangle
    float3 edge1 = MathUtils::float3_subtract(light.v1, light.v0);
    float3 edge2 = MathUtils::float3_subtract(light.v2, light.v0);
    float3 crossProd = MathUtils::cross(edge1, edge2);
    float area = 0.5f * sqrtf(MathUtils::dot(crossProd, crossProd));

    float pdf = (1.0f / numLights) * (1.0f / area);

    // Construct a shadow ray from the hit point toward the light sample
    Ray shadowRay;
    shadowRay.origin = MathUtils::float3_add(hitPoint, MathUtils::float3_scale(normal, 0.001f));
    shadowRay.direction = lightDir;
    if (isOccluded(shadowRay, distance - 0.002f, bvhNodes, triangleIndices, triangles, rootIndex))
        return make_float3(0, 0, 0);

    // Compute the geometric term.
    float factor = (cosTheta * cosThetaLight) / (distance * distance);

    // Direct contribution: L * (BRDF will be applied in the main path tracer) divided by the pdf
    float3 contribution = MathUtils::float3_scale(light.material.emission, factor * (1.0f / pdf));
    return contribution;
}

#endif // DIRECT_LIGHT_SAMPLING_H