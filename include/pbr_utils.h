#ifndef PBR_UTILS_H
#define PBR_UTILS_H

#include "material.h"
#include "math_utils.h"
#include <curand_kernel.h>

// Compute reflection direction, R = I - 2(N * I)N
__device__ __host__ inline float3 reflect(const float3 &I, const float3 &N) {
    return MathUtils::float3_subtract(I, MathUtils::float3_scale(N, 2.0f * MathUtils::dot(I, N)));
}

// Schlick’s approximation for Fresnel factor, a formula that estimates how much light reflects off a surface depending on the viewing angle
__device__ __host__ inline float schlickFresnel(float cosTheta, float3 F0) {
    // For simplicity we assume F0.x = F0.y = F0.z.
    return F0.x + (1.0f - F0.x) * powf(1.0f - cosTheta, 5.0f);
}

#endif // PBR_UTILS_H
