#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <cuda_runtime.h>

namespace MathUtils {

inline __device__ __host__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ __host__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline __device__ __host__ float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    return make_float3(0.0f, 0.0f, 0.0f); // Avoid division by zero
}

inline __device__ __host__ float3 float3_subtract(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ __host__ float3 rotate(const float3 &v, const float3 &targetDir) {
    float3 up = (fabs(targetDir.y) > 0.99f) ? make_float3(1, 0, 0) : make_float3(0, 1, 0);
    float3 right = MathUtils::normalize(MathUtils::cross(up, targetDir));
    float3 newUp = MathUtils::normalize(MathUtils::cross(targetDir, right));

    return make_float3(
        v.x * right.x + v.y * newUp.x + v.z * targetDir.x,
        v.x * right.y + v.y * newUp.y + v.z * targetDir.y,
        v.x * right.z + v.y * newUp.z + v.z * targetDir.z);
}
} // namespace MathUtils

#endif // MATHUTILS_H
