#include "obj_loader.h"
#include "raytracer.h"
#include "math_utils.h"
#include "opengl_utils.h"
#include "bvh.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

// Window dimensions (defaults)
int width = 800;
int height = 600;

// Device pointers for static scene data
Triangle* d_triangles = nullptr;
BVHNode* d_bvhNodes = nullptr;
int* d_triangleIndices = nullptr;
curandState* d_randStates = nullptr;
Triangle* d_lightTriangles = nullptr;  
int d_numLights = 0; 

// Forward declarations
void initDeviceMemory(const vector<Triangle>& h_triangles, const BVH& bvh);
void renderHost(const BVH& bvh);

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./raytracer <path-to-obj> [width height]" << endl;
        return -1;
    }
    // Allow dynamic resolution from command-line arguments
    if (argc >= 4) {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    }

    // Load the OBJ file into a triangle list
    vector<Triangle> h_triangles;
    if (!loadOBJ(argv[1], h_triangles)) {
        cerr << "Failed to load OBJ file!" << endl;
        return -1;
    }

    // Build the BVH for the loaded triangles (only once)
    BVH bvh = buildBVH(h_triangles);

    // Initialize OpenGL (and create the PBO, etc.)
    initOpenGL();
    GLFWwindow* window = glfwGetCurrentContext();

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

// This function initializes device memory for triangles, BVH nodes, triangle indices,
// and random states, and it copies static scene data from host to device once.
void initDeviceMemory(const vector<Triangle>& h_triangles, const BVH& bvh) {
    // Allocate and copy triangles.
    cudaMalloc(&d_triangles, h_triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Allocate and copy BVH nodes.
    size_t nodesSize = bvh.nodes.size() * sizeof(BVHNode);
    cudaMalloc(&d_bvhNodes, nodesSize);
    cudaMemcpy(d_bvhNodes, bvh.nodes.data(), nodesSize, cudaMemcpyHostToDevice);

    // Allocate and copy triangle indices.
    size_t indicesSize = bvh.triangleIndices.size() * sizeof(int);
    cudaMalloc(&d_triangleIndices, indicesSize);
    cudaMemcpy(d_triangleIndices, bvh.triangleIndices.data(), indicesSize, cudaMemcpyHostToDevice);

    // Build list of light triangles from emissive surfaces.
    vector<Triangle> h_lightTriangles;
    for (const auto &tri : h_triangles) {
        if (tri.material.emission.x > 0.0f || tri.material.emission.y > 0.0f || tri.material.emission.z > 0.0f)
            h_lightTriangles.push_back(tri);
    }
    d_numLights = h_lightTriangles.size();
    cudaMalloc(&d_lightTriangles, d_numLights * sizeof(Triangle));
    cudaMemcpy(d_lightTriangles, h_lightTriangles.data(), d_numLights * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Allocate random states.
    cudaMalloc(&d_randStates, width * height * sizeof(curandState));
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    initRandomStates<<<gridSize, blockSize>>>(d_randStates, width, height);
    cudaDeviceSynchronize();
}

// This render function maps the preallocated OpenGL PBO to a CUDA pointer and launches
// the render kernel using the static device data.
void renderHost(const BVH& bvh) {
    uchar4* d_pixels;
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &numBytes, cudaPBOResource);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    renderKernel<<<gridSize, blockSize>>>(d_pixels, width, height,
                                          d_bvhNodes, d_triangleIndices, d_triangles, bvh.rootIndex,
                                          d_randStates, cameraPos, cameraDir, objectYaw, objectPitch,
                                          d_lightTriangles, d_numLights);
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
}