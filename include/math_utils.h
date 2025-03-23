#ifndef MATH_UTILS_H
#define MATH_UTILS_H

/**
 * @file math_utils.h
 * @brief Common vector math operations for float3, used in CUDA-based rendering and geometry processing.
 *
 * Provides device- and host-compatible utility functions for:
 * - Vector arithmetic and algebra (dot, cross, normalize, add, subtract, etc.)
 * - Component-wise min/max and scaling
 * - Coordinate system rotation (yaw/pitch and their inverse)
 *
 * All functions are inlined and suitable for use in both CPU and GPU code.
 */
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265f ///< Pi constant definition if not already defined
#endif

/**
 * @brief Utility functions for vector math operations commonly used in rendering and geometry.
 */
namespace MathUtils {

/**
 * @brief Get the component of a float3 vector based on axis index.
 *
 * @param v: The input float3 vector.
 * @param axis: The axis index (0 = x, 1 = y, 2 = z).
 * @return The value of the selected component.
 */
inline __device__ __host__ float getComponent(const float3 &v, int axis) {
    return (axis == 0) ? v.x : (axis == 1) ? v.y
                                           : v.z;
}

/**
 * @brief Compute the dot product of two float3 vectors.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return The scalar dot product.
 */
inline __device__ __host__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Compute the cross product of two float3 vectors.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return A float3 perpendicular to both input vectors.
 */
inline __device__ __host__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

/**
 * @brief Normalize a float3 vector.
 *
 * @param v: The input vector.
 * @return A unit-length vector in the same direction, or zero vector if input is zero.
 */
inline __device__ __host__ float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    return make_float3(0.0f, 0.0f, 0.0f); // Avoid division by zero
}

/**
 * @brief Subtract two float3 vectors (a - b).
 *
 * @param a: Minuend vector.
 * @param b: Subtrahend vector.
 * @return The resulting float3 difference.
 */
inline __device__ __host__ float3 float3_subtract(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Add two float3 vectors.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return The resulting float3 sum.
 */
inline __device__ __host__ float3 float3_add(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Scale a float3 vector by a scalar.
 *
 * @param v: The input vector.
 * @param scalar: The scalar multiplier.
 * @return The scaled float3 vector.
 */
inline __device__ __host__ float3 float3_scale(const float3 &v, float scalar) {
    return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

/**
 * @brief Multiply two float3 vectors component-wise.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return The resulting float3 with element-wise multiplication.
 */
inline __device__ __host__ float3 float3_multiply(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

/**
 * @brief Compute the component-wise minimum of two float3 vectors.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return A float3 containing the minimum components.
 */
inline __device__ __host__ float3 float3_min(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x),
                       fminf(a.y, b.y),
                       fminf(a.z, b.z));
}

/**
 * @brief Compute the component-wise maximum of two float3 vectors.
 *
 * @param a: First vector.
 * @param b: Second vector.
 * @return A float3 containing the maximum components.
 */
inline __device__ __host__ float3 float3_max(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x),
                       fmaxf(a.y, b.y),
                       fmaxf(a.z, b.z));
}

/**
 * @brief Rotate a vertex in 3D space using yaw and pitch angles.
 *
 * The rotation applies yaw (around the Y-axis) followed by pitch (around the X/Z axis).
 *
 * @param vertex: The position to rotate.
 * @param yaw: Rotation angle around the Y-axis.
 * @param pitch: Rotation angle around the X/Z axis.
 * @return The rotated position.
 */
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

/**
 * @brief Apply inverse rotation using yaw and pitch to transform a vector back to object space.
 *
 * @param v: The vector to rotate back.
 * @param yaw: The original yaw angle used in forward rotation.
 * @param pitch: The original pitch angle used in forward rotation.
 * @return The inverse-rotated vector.
 */
inline __device__ __host__ float3 rotateInverse(const float3 &v, float yaw, float pitch) {
    // Inverse rotation is just negative angles
    float inverseYaw = -yaw;
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

} // namespace MathUtils

#endif // MATH_UTILS_H