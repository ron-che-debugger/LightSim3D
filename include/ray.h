#ifndef RAY_H
#define RAY_H

#include "mathutils.h"
#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const float3 &o, const float3 &d) : origin(o), direction(MathUtils::normalize(d)) {}

    // Compute the ray position at parameter t
    __device__ __host__ float3 at(float t) const {
        return make_float3(origin.x + t * direction.x,
                           origin.y + t * direction.y,
                           origin.z + t * direction.z);
    }
};

#endif