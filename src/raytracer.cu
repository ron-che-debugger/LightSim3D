#include "raytracer.h"

// Inverse-rotate the ray from world space to object (local) space.
__device__ __host__ Ray inverseRotateRay(const Ray& worldRay, float yaw, float pitch)
{
    Ray localRay;
    localRay.origin    = MathUtils::rotateInverse(worldRay.origin,    yaw, pitch);
    localRay.direction = MathUtils::rotateInverse(worldRay.direction, yaw, pitch);
    localRay.direction = MathUtils::normalize(localRay.direction);
    return localRay;
}

// Standard triangle intersection (Möller–Trumbore) in object space.
// Note: Removed per-triangle rotation. Triangles are assumed to be unrotated.
__device__ bool intersectTriangle(const Ray& ray, const Triangle& tri, float& t)
{
    float3 edge1 = MathUtils::float3_subtract(tri.v1, tri.v0);
    float3 edge2 = MathUtils::float3_subtract(tri.v2, tri.v0);
    float3 h = MathUtils::cross(ray.direction, edge2);
    float a = MathUtils::dot(edge1, h);
    if (fabs(a) < 1e-6f) return false;
    
    float f = 1.0f / a;
    float3 s = MathUtils::float3_subtract(ray.origin, tri.v0);
    float u = f * MathUtils::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    
    float3 q = MathUtils::cross(s, edge1);
    float v = f * MathUtils::dot(ray.direction, q);
    if (v < 0.0f || (u + v) > 1.0f) return false;
    
    t = f * MathUtils::dot(edge2, q);
    return (t > 1e-6f);
}

// AABB intersection using the slab method. Assumes the ray is in object space.
__device__ bool intersectAABB(const Ray &ray, AABB box, float t_min, float t_max)
{
    for (int i = 0; i < 3; i++) {
        float origin, dir, minVal, maxVal;
        if (i == 0) {
            origin = ray.origin.x;
            dir = ray.direction.x;
            minVal = box.min.x;
            maxVal = box.max.x;
        } else if (i == 1) {
            origin = ray.origin.y;
            dir = ray.direction.y;
            minVal = box.min.y;
            maxVal = box.max.y;
        } else {
            origin = ray.origin.z;
            dir = ray.direction.z;
            minVal = box.min.z;
            maxVal = box.max.z;
        }
        float invD = 1.0f / dir;
        float t0 = (minVal - origin) * invD;
        float t1 = (maxVal - origin) * invD;
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;
}

// Traverse the BVH in object space. The ray passed in is already inverse-rotated.
__device__ bool traverseBVH(const Ray &ray, BVHNode* bvhNodes, int* triangleIndices,
                             Triangle* triangles, int rootIndex,
                             float &closestT, Triangle &hitTriangle)
{
    bool hit = false;
    const int stackSize = 64;
    int stack[stackSize];
    int stackPtr = 0;
    stack[stackPtr++] = rootIndex;
    
    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        BVHNode node = bvhNodes[nodeIndex];
        if (!intersectAABB(ray, node.bbox, 0.001f, closestT))
            continue;
        if (node.isLeaf) {
            for (int i = node.start; i < node.start + node.count; i++) {
                int triIndex = triangleIndices[i];
                float t;
                if (intersectTriangle(ray, triangles[triIndex], t) && t < closestT) {
                    closestT = t;
                    hitTriangle = triangles[triIndex];
                    hit = true;
                }
            }
        } else {
            if (node.left != -1) stack[stackPtr++] = node.left;
            if (node.right != -1) stack[stackPtr++] = node.right;
        }
    }
    return hit;
}

__global__ void initRandomStates(curandState* randStates, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(1234, idx, 0, &randStates[idx]);
}

__device__ float3 randomCosineWeightedHemisphere(float3 normal, curandState* randState)
{
    float r1 = curand_uniform(randState);
    float r2 = curand_uniform(randState);
    
    float theta = acosf(sqrtf(1.0f - r1));
    float phi = 2.0f * M_PI * r2;
    
    float x = sinf(theta) * cosf(phi);
    float y = sinf(theta) * sinf(phi);
    float z = cosf(theta);
    
    float3 randomVec = make_float3(x, y, z);
    if (MathUtils::dot(randomVec, normal) < 0.0f)
        randomVec = MathUtils::float3_scale(randomVec, -1.0f);
    
    return randomVec;
}

// Path tracing routine that operates in object space.
__device__ float3 pathTrace(Ray ray, BVHNode* bvhNodes, int* triangleIndices, Triangle* triangles,
                            int rootIndex, curandState* randState, int depth)
{
    float3 color = make_float3(0, 0, 0);
    float3 throughput = make_float3(1, 1, 1);
    
    for (int i = 0; i < depth; i++) {
        float closestT = 1e20f;
        Triangle hitTriangle;
        bool hit = traverseBVH(ray, bvhNodes, triangleIndices, triangles, rootIndex, closestT, hitTriangle);
        if (!hit)
            break;
        
        float3 hitPoint = MathUtils::float3_add(ray.origin, MathUtils::float3_scale(ray.direction, closestT));
        float3 normal = hitTriangle.normal;
        
        // Check for a light source (assuming emission indicates a light).
        if (MathUtils::dot(normal, ray.direction) < 0.0f) {
            color = MathUtils::float3_add(color, MathUtils::float3_multiply(throughput, hitTriangle.emission));
            break;
        }
        
        float3 newDir = randomCosineWeightedHemisphere(normal, randState);
        throughput = MathUtils::float3_multiply(throughput, make_float3(0.8f, 0.8f, 0.8f));
        throughput = MathUtils::float3_scale(throughput, MathUtils::dot(normal, newDir));
        
        ray.origin = hitPoint;
        ray.direction = newDir;
        
        float prob = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (curand_uniform(randState) > prob)
            break;
        throughput = MathUtils::float3_scale(throughput, 1.0f / prob);
    }
    return color;
}

// Render kernel: transform the world-space ray into object space before tracing.
__global__ void renderKernel(uchar4* pixels, int width, int height,
    BVHNode* bvhNodes, int* triangleIndices, Triangle* triangles, int rootIndex,
    curandState* randStates,
    float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    curandState localRandState = randStates[idx];
    float3 color = make_float3(0, 0, 0);
    int numSamples = 100;
    
    for (int i = 0; i < numSamples; i++) {
        // Construct a world-space ray from the camera.
        Ray worldRay;
        worldRay.origin = cameraPos;
        
        // Apply anti-aliasing jitter.
        float jitterX = curand_uniform(&localRandState) - 0.5f;
        float jitterY = curand_uniform(&localRandState) - 0.5f;
        float aspectRatio = (float)width / (float)height;
        float3 pixelDir = MathUtils::normalize(make_float3(
            ((x + jitterX - width / 2.0f) / width) * aspectRatio * 2.0f,
            ((y + jitterY - height / 2.0f) / height) * 2.0f,
            1.0f));
        worldRay.direction = pixelDir;
        
        // Inverse-rotate the world ray into object (local) space.
        Ray localRay = inverseRotateRay(worldRay, objectYaw, objectPitch);
        
        color = MathUtils::float3_add(color,
            pathTrace(localRay, bvhNodes, triangleIndices, triangles, rootIndex, &localRandState, 5));
    }
    
    color = MathUtils::float3_scale(color, 1.0f / numSamples);
    pixels[idx] = make_uchar4(
        fminf(color.x * 255, 255),
        fminf(color.y * 255, 255),
        fminf(color.z * 255, 255),
        255);
}