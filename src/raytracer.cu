#include "raytracer.h"
#include "direct_light_sampling.h"

/**
 * @brief Transform a world-space ray into object (local) space using inverse rotation.
 *
 * @param worldRay The ray in world coordinates.
 * @param yaw Yaw angle used to inverse-rotate.
 * @param pitch Pitch angle used to inverse-rotate.
 * @return The ray transformed to object space.
 */
__device__ __host__ Ray inverseRotateRay(const Ray& worldRay, float yaw, float pitch)
{
    Ray localRay;
    localRay.origin    = MathUtils::rotateInverse(worldRay.origin,    yaw, pitch);
    localRay.direction = MathUtils::rotateInverse(worldRay.direction, yaw, pitch);
    localRay.direction = MathUtils::normalize(localRay.direction);
    return localRay;
}

/**
 * @brief Test intersection between a ray and a triangle using Möller–Trumbore algorithm.
 *
 * @param ray The ray to test.
 * @param tri The triangle.
 * @param t Output hit distance.
 * @return True if the ray hits the triangle.
 */
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

/**
 * @brief Test intersection between a ray and an AABB using the slab method.
 *
 * @param ray The ray in object space.
 * @param box The bounding box.
 * @param t_min Minimum allowed t.
 * @param t_max Maximum allowed t.
 * @return True if the ray intersects the box.
 */
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

        // Computes t0 and t1, which are entry and exit points of the ray along the current axis
        float invD = 1.0f / dir;
        float t0 = (minVal - origin) * invD;
        float t1 = (maxVal - origin) * invD;

        // If the ray direction is negative, t0 and t1 need to be swapped
        // Ensures that t0 is always the entry point and t1 is the exit point
        if (invD < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        // Shrinks the valid intersection range
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;

        // This means the valid interval collapsed and there’s no overlap
        if (t_max <= t_min)
            return false;
    }
    return true;
}

/**
 * @brief Traverse the BVH to find the closest triangle hit by a ray.
 *
 * @param ray Ray in object space.
 * @param bvhNodes Pointer to BVH nodes.
 * @param triangleIndices Triangle index lookup.
 * @param triangles Scene geometry.
 * @param rootIndex Index of BVH root node.
 * @param closestT Output closest hit distance.
 * @param hitTriangle Output triangle that was hit.
 * @return True if any triangle was hit.
 */
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

/**
 * @brief Initializes CURAND random states for each pixel.
 *
 * @param randStates curandState array.
 * @param width Image width.
 * @param height Image height.
 */
__global__ void initRandomStates(curandState* randStates, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(1234, idx, 0, &randStates[idx]);
}

/**
 * @brief Sample a cosine-weighted direction in the hemisphere around a surface normal.
 *
 * @param normal The surface normal.
 * @param randState Random generator.
 * @return A new sampled direction.
 */
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

/**
 * @brief Recursive path tracer using simple PBR shading and next-event estimation.
 *
 * This integrator handles both diffuse and specular reflection based on the material's metallic property.
 *
 * - For matte (diffuse) surfaces, next-event estimation is used by directly sampling a light source.
 * - For metallic (specular) surfaces, perfect mirror reflection is used with Fresnel scaling (Schlick's approximation).
 *
 * Russian roulette is applied after each bounce to probabilistically terminate low-contribution paths.
 *
 * @param ray Ray to trace (should be in object space).
 * @param bvhNodes BVH structure.
 * @param triangleIndices Triangle index buffer.
 * @param triangles Scene geometry.
 * @param rootIndex Index of root node in BVH.
 * @param randState Random state for sampling.
 * @param depth Maximum bounce depth.
 * @param lightTriangles Emissive triangles in the scene.
 * @param numLights Number of emissive triangles.
 * @return Computed radiance for the input ray.
 */
__device__ float3 pathTrace(
    Ray ray,
    BVHNode* bvhNodes, int* triangleIndices, Triangle* triangles, int rootIndex,
    curandState* randState, int depth,
    Triangle* lightTriangles, int numLights)
{
    float3 color = make_float3(0, 0, 0);
    float3 throughput = make_float3(1, 1, 1);
    
    for (int i = 0; i < depth; i++) {
        float closestT = 1e20f;
        Triangle hitTriangle;
        bool hit = traverseBVH(ray, bvhNodes, triangleIndices, triangles, rootIndex, closestT, hitTriangle);
        
        /** Unnecessary since we have an environment sphere, each ray hits something eventually
         * if (!hit) {
            color = MathUtils::float3_multiply(throughput, make_float3(0.5f, 0.7f, 1.0f)); // Uniform sky blue 
            break;
        }
        */
        
        float3 hitPoint = MathUtils::float3_add(ray.origin, MathUtils::float3_scale(ray.direction, closestT));
        float3 normal = hitTriangle.normal;
        
        // If the hit surface is emissive, add its contribution and terminate.
        if (hitTriangle.material.emission.x > 0.0f ||
            hitTriangle.material.emission.y > 0.0f ||
            hitTriangle.material.emission.z > 0.0f) {

            if (i == 0) {
                // For primary rays hitting the environment, override to sky blue.
                color = MathUtils::float3_multiply(throughput, make_float3(0.5f, 0.7f, 1.0f));
            } else {
                // For reflective bounces, use the environment's actual emission.
                color = MathUtils::float3_multiply(throughput, hitTriangle.material.emission);
            }
            break;
        }
        
        // Decide between diffuse and specular reflection.
        float sample = curand_uniform(randState);
        float diffuseProbability = 1.0f - hitTriangle.material.metallic;
        float3 newDir;
        if (sample < diffuseProbability) {
            // Direct lighting (next–event estimation)
            float3 directLight = sampleDirectLight(hitPoint, normal,
                bvhNodes, triangleIndices, triangles, rootIndex,
                lightTriangles, numLights, randState);

            // Diffuse BRDF = albedo/π.
            float3 brdf = MathUtils::float3_scale(hitTriangle.material.albedo, 1.0f / M_PI);
            color = MathUtils::float3_add(color,
                        MathUtils::float3_multiply(throughput,
                        MathUtils::float3_multiply(brdf, directLight)));

            // Diffuse bounce: sample a new direction.
            newDir = randomCosineWeightedHemisphere(normal, randState);
            throughput = MathUtils::float3_multiply(throughput, brdf);
        } else {
            // Specular reflection: mirror reflection.
            newDir = reflect(MathUtils::float3_scale(ray.direction, -1.0f), normal);

            // Compute Fresnel reflectance (Schlick's approximation)
            float R0 = 0.04f; // Base reflectance for non-metal
            float3 F0 = make_float3(R0, R0, R0);
            F0 = MathUtils::float3_add(MathUtils::float3_scale(F0, 1.0f - hitTriangle.material.metallic),
                                       MathUtils::float3_scale(hitTriangle.material.albedo, hitTriangle.material.metallic));
            float fresnel = schlickFresnel(fmaxf(MathUtils::dot(normal, newDir), 0.0f), F0);

            
            throughput = MathUtils::float3_multiply(throughput, make_float3(fresnel, fresnel, fresnel));
        }
        
        ray.origin = hitPoint;
        ray.direction = newDir;
        
        // Russian roulette termination.
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (curand_uniform(randState) > p)
            break;
        throughput = MathUtils::float3_scale(throughput, 1.0f / p);
    }
    return color;
}

/**
 * @brief CUDA kernel that launches the path tracer for each pixel, handles camera orientation and anti-aliasing.
 *
 * @param pixels Output framebuffer.
 * @param width Image width.
 * @param height Image height.
 * @param bvhNodes BVH node array.
 * @param triangleIndices Triangle index array.
 * @param triangles Geometry list.
 * @param rootIndex Root BVH node index.
 * @param randStates Per-pixel CURAND states.
 * @param cameraPos World-space camera position.
 * @param cameraDir Camera direction (unused here, but part of the interface).
 * @param objectYaw Yaw rotation applied to scene.
 * @param objectPitch Pitch rotation applied to scene.
 * @param lightTriangles List of emissive triangles.
 * @param numLights Number of lights.
 */
__global__ void renderKernel(uchar4* pixels, int width, int height,
    BVHNode* bvhNodes, int* triangleIndices, Triangle* triangles, int rootIndex,
    curandState* randStates,
    float3 cameraPos, float3 cameraDir, float objectYaw, float objectPitch,
    Triangle* lightTriangles, int numLights)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    curandState localRandState = randStates[idx];
    float3 color = make_float3(0, 0, 0);
    int numSamples = 40;
    
    for (int i = 0; i < numSamples; i++) {
        Ray worldRay;
        worldRay.origin = cameraPos;
        
        // Anti-aliasing jitter.
        float jitterX = curand_uniform(&localRandState) - 0.5f;
        float jitterY = curand_uniform(&localRandState) - 0.5f;
        float aspectRatio = (float)width / (float)height;
        float3 pixelDir = MathUtils::normalize(make_float3(
            ((x + jitterX - width / 2.0f) / width) * aspectRatio * 2.0f,
            ((y + jitterY - height / 2.0f) / height) * 2.0f,
            1.0f));
        worldRay.direction = pixelDir;
        
        // Inverse-rotate the ray into object (local) space.
        Ray localRay = inverseRotateRay(worldRay, objectYaw, objectPitch);
        
        color = MathUtils::float3_add(color,
            pathTrace(localRay, bvhNodes, triangleIndices, triangles, rootIndex,
                      &localRandState, 20, lightTriangles, numLights));
    }
    
    color = MathUtils::float3_scale(color, 1.0f / numSamples);
    pixels[idx] = make_uchar4(
        fminf(color.x * 255, 255),
        fminf(color.y * 255, 255),
        fminf(color.z * 255, 255),
        255);
}