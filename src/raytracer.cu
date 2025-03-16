#include <cuda_runtime.h>
#include "mathutils.h"
#include "obj_loader.h"
#include "ray.h"

using namespace std;

__device__ float3 computeLighting(const float3& normal, const float3& lightDir) {
    float intensity = fmaxf(MathUtils::dot(normal, lightDir), 0.0f);
    
    // Simple Lambertian shading
    // The brightness of a surface depends on how much light directly hits it
    // A surface is brightest when facing the light directly, and darkest when perpendicular to the light
    return make_float3(intensity, intensity, intensity); // Simple Lambertian shading
}

__device__ bool intersectTriangle(const Ray& ray, const Triangle& tri, float& t){
    float3 edge1 = MathUtils::float3_subtract(tri.v1, tri.v0);
    float3 edge2 = MathUtils::float3_subtract(tri.v2, tri.v0);
    float3 h = MathUtils::cross(ray.direction, edge2);
    
    // Detects parallel ray
    float a = MathUtils::dot(edge1, h);
    if (fabs(a) < 1e-6){
        return false;
    }

    // Compute barycentric coordinate u
    // Any points inside the triangle can be represented by the barycentric coordinates (u, v) by:
    // P = v0 + u * (v1 - v0) + v * (v2 - v0)
    float f = 1.0f / a;  // Precompute inverse determinant for efficiency
    float3 s = MathUtils::float3_subtract(ray.origin, tri.v0);  // Vector from triangle vertex v0 to ray origin
    float u = f * MathUtils::dot(s, h);  // Compute barycentric coordinate u (weight for edge1)
    if (u < 0.0 || u > 1.0){ // Reject if u is out of bounds (outside triangle)
        return false;
    }

    // Compute barycentric coordinate v
    float3 q = MathUtils::cross(s, edge1);  // Compute perpendicular vector to edge1
    float v = f * MathUtils::dot(ray.direction, q);  // Compute barycentric coordinate v (weight for edge2)
    if (v < 0.0 || u + v > 1.0){ // Reject if v is out of bounds or u + v > 1 (outside triangle)
        return false; 
    }

    // t tells us if the intersection is in front of the ray origin
    // If t > 0, the intersection is in the direction the ray is traveling
    // If t < 0, the intersection is behind the ray origin, meaning the camera "sees nothing"
    t = f * MathUtils::dot(edge2, q);  // Compute intersection distance along the ray
    return (t > 1e-6);  // Accept intersection if t is positive (ensures ray is moving forward)
}

__global__ void renderKernel(uchar4* pixels, int width, int height, Triangle* triangles, int numTriangles, float3 cameraPos, float3 cameraDir){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height){
        return;
    }

    int idx = y * width + x;

    Ray ray;

    // In a typical 3D setup, the XY plane represents the screen, and the camera is positioned along the negative Z-axis
    ray.origin = cameraPos;

    float3 pixelDir = MathUtils::normalize(make_float3((x - width / 2) * 0.002f, (y - height / 2) * 0.002f, 1.0f));

    ray.direction = MathUtils::rotate(pixelDir, cameraDir);

    float minT = 1e20;
    float3 color = make_float3(0, 0, 0); // no intersection = black pixel.
    float3 lightDir = MathUtils::normalize(make_float3(1, 1, -1)); // (right, up, back)

    for (int i = 0; i < numTriangles; ++i){
        float t;

        // Only render the closest triangles to the camera
        if (intersectTriangle(ray, triangles[i], t) && t < minT){
            minT = t;
            color = computeLighting(triangles[i].normal, lightDir);
        }
    }

    pixels[idx] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}