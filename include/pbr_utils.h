#ifndef PBR_UTILS_H
#define PBR_UTILS_H

#include "material.h"
#include "math_utils.h"
#include <curand_kernel.h>

/**
 * @brief Compute the reflection direction based on an incident vector and a normal.
 *
 * Uses the formula: R = I - 2 * dot(N, I) * N
 *
 * @param I Incident direction vector.
 * @param N Surface normal vector.
 * @return Reflected direction vector.
 */
__device__ __host__ inline float3 reflect(const float3 &I, const float3 &N) {
    return MathUtils::float3_subtract(I, MathUtils::float3_scale(N, 2.0f * MathUtils::dot(I, N)));
}

/**
 * @brief Compute the Fresnel reflectance using Schlick's approximation.
 *
 * This formula estimates how much light reflects off a surface depending on the angle
 * between the view direction and the surface normal.
 *
 * @param cosTheta Cosine of the angle between view direction and surface normal.
 * @param F0 Reflectance at normal incidence (assumed to be the same for all channels).
 * @return Approximate Fresnel reflection coefficient.
 */
__device__ __host__ inline float schlickFresnel(float cosTheta, float3 F0) {
    // For simplicity we assume F0.x = F0.y = F0.z.
    return F0.x + (1.0f - F0.x) * powf(1.0f - cosTheta, 5.0f);
}

#endif // PBR_UTILS_H