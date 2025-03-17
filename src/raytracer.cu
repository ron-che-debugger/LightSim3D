#include <cuda_runtime.h>
#include "math_utils.h"
#include "obj_loader.h"
#include "ray.h"

__device__ float3 computeLighting(const float3& normal, const float3& lightDir) {
    float intensity = fmaxf(MathUtils::dot(normal, lightDir), 0.0f);
    return make_float3(intensity, intensity, intensity);
}

__device__ bool intersectTriangle(const Ray& ray, const Triangle& tri, float& t){
    float3 edge1 = MathUtils::float3_subtract(tri.v1, tri.v0);
    float3 edge2 = MathUtils::float3_subtract(tri.v2, tri.v0);
    float3 h = MathUtils::cross(ray.direction, edge2);
    float a = MathUtils::dot(edge1, h);
    if (fabs(a) < 1e-6) return false;

    float f = 1.0f / a;
    float3 s = MathUtils::float3_subtract(ray.origin, tri.v0);
    float u = f * MathUtils::dot(s, h);
    if (u < 0.0 || u > 1.0) return false;

    float3 q = MathUtils::cross(s, edge1);
    float v = f * MathUtils::dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) return false;

    t = f * MathUtils::dot(edge2, q);
    return (t > 1e-6);
}

__global__ void renderKernel(uchar4* pixels, int width, int height, Triangle* triangles, int numTriangles, float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Ray ray;
    ray.origin = cameraPos;

    float3 pixelDir = MathUtils::normalize(make_float3((x - width / 2) * 0.002f, (y - height / 2) * 0.002f, 1.0f));
    ray.direction = pixelDir;

    float minT = 1e20;
    float3 color = make_float3(0, 0, 0);
    float3 lightDir = MathUtils::normalize(make_float3(1, 1, -1));

    for (int i = 0; i < numTriangles; ++i) {
        float t;
        Triangle tri = triangles[i];

        // Rotate triangle before intersection test
        tri.v0 = MathUtils::rotateObject(tri.v0, objectYaw, objectPitch);
        tri.v1 = MathUtils::rotateObject(tri.v1, objectYaw, objectPitch);
        tri.v2 = MathUtils::rotateObject(tri.v2, objectYaw, objectPitch);

        if (intersectTriangle(ray, tri, t) && t < minT) {
            minT = t;
            color = computeLighting(tri.normal, lightDir);
        }
    }

    pixels[idx] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}
