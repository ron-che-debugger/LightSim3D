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

inline __device__ __host__ float3 float3_add(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ __host__ float3 float3_scale(const float3& v, float scalar) {
    return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

// Rotation Matrix Function
inline __device__ __host__ float3 rotateObject(float3 vertex, float yaw, float pitch) {
    float cosYaw = cosf(yaw);
    float sinYaw = sinf(yaw);
    float cosPitch = cosf(pitch);
    float sinPitch = sinf(pitch);

    float3 rotated;
    
    rotated.x = cosYaw * vertex.x - sinYaw * vertex.z;
    rotated.z = sinYaw * vertex.x + cosYaw * vertex.z;
    rotated.y = cosPitch * vertex.y - sinPitch * rotated.z;
    rotated.z = sinPitch * vertex.y + cosPitch * rotated.z;

    return rotated;
}

} // namespace MathUtils

#endif // MATHUTILS_H
