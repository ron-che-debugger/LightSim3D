#include "bvh.h"
#include "math_utils.h"
#include "obj_loader.h"
#include "opengl_utils.h"
#include "raytracer.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// Window dimensions (defaults)
int width = 800;
int height = 600;

// Device pointers for static scene data
Triangle *d_triangles = nullptr;
BVHNode *d_bvhNodes = nullptr;
int *d_triangleIndices = nullptr;
curandState *d_randStates = nullptr;
Triangle *d_lightTriangles = nullptr;
int d_numLights = 0;

// Forward declarations
void initDeviceMemory(const vector<Triangle> &h_triangles, const BVH &bvh);
vector<Triangle> createEnvironmentSphere(float radius, int rings, int sectors, float3 emission, float3 albedo);
void renderHost(const BVH &bvh);

/**
 * @brief Entry point for the CUDA path tracer.
 *
 * This function parses arguments, loads a 3D model from OBJ, applies material effects,
 * constructs a BVH, sets up OpenGL and CUDA interop, and enters the render loop.
 *
 * @param argc Argument count.
 * @param argv Argument values. Usage: ./raytracer <obj_file> [width height] [effect]
 * @return Exit code (0 on success, -1 on failure).
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: ./raytracer <path-to-obj> [width height]" << endl;
        return -1;
    }
    // Allow dynamic resolution from command-line arguments
    if (argc >= 4) {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    }

    // Parse the optional rendering effect argument.
    // Supported effects: "metal", "matte", or "default" (if unspecified)
    string effect = "default";
    if (argc >= 5) {
        effect = argv[4];
    }
    cout << "Using rendering effect: " << effect << endl;

    // Load the OBJ file into a triangle list.
    vector<Triangle> h_triangles;
    if (!loadOBJ(argv[1], h_triangles)) {
        cerr << "Failed to load OBJ file!" << endl;
        return -1;
    }

    // Apply the chosen rendering effect to object materials.
    applyRenderingEffect(h_triangles, effect);

    // Create environment geometry (a large sphere that encloses the scene)
    float envRadius = 100.0f;
    int envRings = 2;
    int envSectors = 4;
    float3 envEmission = make_float3(1.0, 1.0f, 1.0f); // Emission intensity/color for the environment
    float3 envAlbedo = make_float3(1.0f, 1.0f, 1.0f);
    vector<Triangle> envTriangles = createEnvironmentSphere(envRadius, envRings, envSectors, envEmission, envAlbedo);

    // Append environment triangles to the scene
    h_triangles.insert(h_triangles.end(), envTriangles.begin(), envTriangles.end());

    // Build the BVH for the loaded triangles (only once)
    BVH bvh = buildBVH(h_triangles);

    // Initialize OpenGL (and create the PBO, etc.)
    initOpenGL();
    GLFWwindow *window = glfwGetCurrentContext();

    // Allocate and initialize all static device memory once
    initDeviceMemory(h_triangles, bvh);

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        updateCamera(window);

        // Render using preallocated static scene data
        renderHost(bvh);

        // Draw the result by mapping the PBO to the screen
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        drawScreen();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup device memory
    cudaFree(d_triangles);
    cudaFree(d_bvhNodes);
    cudaFree(d_triangleIndices);
    cudaFree(d_randStates);

    glDeleteBuffers(1, &pbo);
    cudaGraphicsUnregisterResource(cudaPBOResource);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

/**
 * @brief Initializes all device memory used during rendering, including:
 *        - Copying triangle geometry to device memory
 *        - Building the BVH node buffer
 *        - Copying triangle indices
 *        - Extracting emissive light triangles
 *        - Allocating per-pixel CURAND random states
 *
 * @param h_triangles Host-side triangle geometry.
 * @param bvh The precomputed BVH structure.
 */
void initDeviceMemory(const vector<Triangle> &h_triangles, const BVH &bvh) {
    // Allocate and copy triangles.
    cudaMalloc(&d_triangles, h_triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Allocate and copy BVH nodes
    size_t nodesSize = bvh.nodes.size() * sizeof(BVHNode);
    cudaMalloc(&d_bvhNodes, nodesSize);
    cudaMemcpy(d_bvhNodes, bvh.nodes.data(), nodesSize, cudaMemcpyHostToDevice);

    // Allocate and copy triangle indices
    size_t indicesSize = bvh.triangleIndices.size() * sizeof(int);
    cudaMalloc(&d_triangleIndices, indicesSize);
    cudaMemcpy(d_triangleIndices, bvh.triangleIndices.data(), indicesSize, cudaMemcpyHostToDevice);

    // Build list of light triangles from emissive surfaces
    vector<Triangle> h_lightTriangles;
    for (const auto &tri : h_triangles) {
        if (tri.material.emission.x > 0.0f || tri.material.emission.y > 0.0f || tri.material.emission.z > 0.0f)
            h_lightTriangles.push_back(tri);
    }
    d_numLights = h_lightTriangles.size();
    cudaMalloc(&d_lightTriangles, d_numLights * sizeof(Triangle));
    cudaMemcpy(d_lightTriangles, h_lightTriangles.data(), d_numLights * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Allocate random states
    cudaMalloc(&d_randStates, width * height * sizeof(curandState));
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    initRandomStates<<<gridSize, blockSize>>>(d_randStates, width, height);
    cudaDeviceSynchronize();
}

/**
 * @brief Creates an emissive environment sphere around the scene using a UV sphere approximation.
 *
 * The sphere is composed of multiple triangles and acts as both a light source and a fallback
 * background surface for rays that don't hit any object geometry.
 *
 * @param radius Radius of the environment sphere.
 * @param rings Number of horizontal subdivisions.
 * @param sectors Number of vertical subdivisions.
 * @param emission Emissive color of the environment surface.
 * @param albedo Diffuse color of the environment surface.
 * @return A vector of triangles representing the environment geometry.
 */
vector<Triangle> createEnvironmentSphere(float radius, int rings, int sectors, float3 emission, float3 albedo) {
    vector<Triangle> sphereTriangles;
    vector<float3> vertices;

    // Generate vertices for a UV sphere
    for (int i = 0; i <= rings; i++) {             // Horizontal divisions
        float theta = i * M_PI / rings;            // [0, pi]
        for (int j = 0; j <= sectors; j++) {       // Vertical divisions
            float phi = j * 2.0f * M_PI / sectors; // [0, 2pi]
            float x = radius * sinf(theta) * cosf(phi);
            float y = radius * cosf(theta);
            float z = radius * sinf(theta) * sinf(phi);
            vertices.push_back(make_float3(x, y, z));
        }
    }

    // Create triangles for each quad on the sphere surface
    for (int i = 0; i < rings; i++) {
        for (int j = 0; j < sectors; j++) {
            int first = i * (sectors + 1) + j;
            int second = first + sectors + 1;

            // Triangle 1
            Triangle t1;
            t1.v0 = vertices[first];
            t1.v1 = vertices[second];
            t1.v2 = vertices[first + 1];
            float3 edge1 = MathUtils::float3_subtract(t1.v1, t1.v0);
            float3 edge2 = MathUtils::float3_subtract(t1.v2, t1.v0);
            float3 n = MathUtils::normalize(MathUtils::cross(edge1, edge2));
            // Invert the normal if it points outward
            if (MathUtils::dot(n, t1.v0) > 0)
                n = MathUtils::float3_scale(n, -1.0f);
            t1.normal = n;
            t1.material.albedo = albedo;
            t1.material.emission = emission;
            t1.material.metallic = 0.0f;
            t1.isEnvironment = true;
            sphereTriangles.push_back(t1);

            // Triangle 2
            Triangle t2;
            t2.v0 = vertices[second];
            t2.v1 = vertices[second + 1];
            t2.v2 = vertices[first + 1];
            float3 edge1b = MathUtils::float3_subtract(t2.v1, t2.v0);
            float3 edge2b = MathUtils::float3_subtract(t2.v2, t2.v0);
            float3 n2 = MathUtils::normalize(MathUtils::cross(edge1b, edge2b));
            if (MathUtils::dot(n2, t2.v0) > 0)
                n2 = MathUtils::float3_scale(n2, -1.0f);
            t2.normal = n2;
            t2.material.albedo = albedo;
            t2.material.emission = emission;
            t2.material.metallic = 0.0f;
            t2.isEnvironment = true;
            sphereTriangles.push_back(t2);
        }
    }
    return sphereTriangles;
}

/**
 * @brief Host-side render function that maps the CUDA-OpenGL PBO, launches the render kernel,
 *        and unmaps the buffer. The kernel performs full path tracing using static device data.
 *
 * @param bvh The scene's bounding volume hierarchy used for acceleration.
 */
void renderHost(const BVH &bvh) {
    uchar4 *d_pixels;
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pixels, &numBytes, cudaPBOResource);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    renderKernel<<<gridSize, blockSize>>>(d_pixels, width, height,
                                          d_bvhNodes, d_triangleIndices, d_triangles, bvh.rootIndex,
                                          d_randStates, cameraPos, cameraDir, objectYaw, objectPitch,
                                          d_lightTriangles, d_numLights);
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
}