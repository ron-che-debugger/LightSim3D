#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265f
#endif

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

inline __device__ __host__ float3 float3_scale(const float3 &v, float scalar) {
    return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

inline __device__ __host__ float3 float3_multiply(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ __host__ float3 float3_min(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), 
                       fminf(a.y, b.y), 
                       fminf(a.z, b.z));
}

inline __device__ __host__ float3 float3_max(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), 
                       fmaxf(a.y, b.y), 
                       fmaxf(a.z, b.z));
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

inline __device__ __host__ float3 rotateInverse(const float3& v, float yaw, float pitch)
{
    // Inverse rotation is just negative angles
    float inverseYaw   = -yaw;
    float inversePitch = -pitch;

    // Apply pitch first (around X-axis or a chosen convention)
    float cosP = cosf(inversePitch);
    float sinP = sinf(inversePitch);
    float3 tmp = make_float3(v.x, cosP * v.y - sinP * v.z, sinP * v.y + cosP * v.z);

    // Then apply yaw (around Y-axis)
    float cosY = cosf(inverseYaw);
    float sinY = sinf(inverseYaw);
    float3 rotated;
    rotated.x = cosY * tmp.x + sinY * tmp.z;
    rotated.y = tmp.y;
    rotated.z = -sinY * tmp.x + cosY * tmp.z;

    return rotated;
}

inline __device__ __host__ float getComponent(const float3& v, int axis) {
    return (axis == 0) ? v.x : (axis == 1) ? v.y : v.z;
}

} // namespace MathUtils

#endif // MATH_UTILS_H
