#ifndef RAY_H
#define RAY_H

#include "math_utils.h"
#include <cuda_runtime.h>

/**
 * @brief Represents a ray in 3D space with an origin and a direction.
 *
 * Used for ray tracing calculations, such as intersection tests and shading.
 */
struct Ray {
    float3 origin;    /// Starting point of the ray
    float3 direction; /// Normalized direction vector

    /**
     * @brief Default constructor. Leaves members uninitialized.
     */
    __host__ __device__ Ray() {}

    /**
     * @brief Construct a ray with the given origin and direction.
     *
     * @param o Origin of the ray.
     * @param d Direction of the ray (automatically normalized).
     */
    __host__ __device__ Ray(const float3 &o, const float3 &d) : origin(o), direction(MathUtils::normalize(d)) {}

    /**
     * @brief Compute a position along the ray at parameter t.
     *
     * @param t Distance along the ray.
     * @return The 3D point at origin + t * direction.
     */
    __device__ __host__ float3 at(float t) const {
        return make_float3(origin.x + t * direction.x,
                           origin.y + t * direction.y,
                           origin.z + t * direction.z);
    }
};

#endif