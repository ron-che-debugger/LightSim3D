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

using namespace std;

// Window dimensions
int width = 800;   // Define the image width (default)
int height = 600;  // Define the image height (default)

// In renderHost, after loading h_triangles:
void renderHost(const std::vector<Triangle>& h_triangles) {
    // Build BVH from the triangle list
    BVH bvh = buildBVH(h_triangles);
    
    // Allocate and copy triangles to device memory
    Triangle* d_triangles;
    cudaMalloc(&d_triangles, h_triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    
    // Allocate and copy BVH nodes to device memory
    BVHNode* d_bvhNodes;
    size_t nodesSize = bvh.nodes.size() * sizeof(BVHNode);
    cudaMalloc(&d_bvhNodes, nodesSize);
    cudaMemcpy(d_bvhNodes, bvh.nodes.data(), nodesSize, cudaMemcpyHostToDevice);
    
    // Allocate and copy triangle indices (for leaves) to device memory
    int* d_triangleIndices;
    size_t indicesSize = bvh.triangleIndices.size() * sizeof(int);
    cudaMalloc(&d_triangleIndices, indicesSize);
    cudaMemcpy(d_triangleIndices, bvh.triangleIndices.data(), indicesSize, cudaMemcpyHostToDevice);
    
    // Lock OpenGL PBO and map it to a CUDA pointer (unchanged)
    uchar4* d_pixels;
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &numBytes, cudaPBOResource);
    
    // Allocate and initialize random states (unchanged)
    curandState* d_randStates;
    cudaMalloc(&d_randStates, width * height * sizeof(curandState));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    initRandomStates<<<gridSize, blockSize>>>(d_randStates, width, height);
    cudaDeviceSynchronize();
    
    // Launch kernel with BVH parameters:
    renderKernel<<<gridSize, blockSize>>>(d_pixels, width, height,
                                          d_bvhNodes, d_triangleIndices, d_triangles, bvh.rootIndex,
                                          d_randStates, cameraPos, cameraDir, objectYaw, objectPitch);
    
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
    
    // Free device memory used for BVH and triangles
    cudaFree(d_triangles);
    cudaFree(d_bvhNodes);
    cudaFree(d_triangleIndices);
    cudaFree(d_randStates);
}


void drawScreen() {
    glClear(GL_COLOR_BUFFER_BIT); // Reset the screen
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0); // Draw pixel data to the screen by pulling it from the currently bound PBO
}

int main(int argc, char** argv){
    if (argc < 2){
        cerr << "Usage: ./raytracer <path-to-obj> [width height]" << endl;
        return -1;
    }

    // Allow dynamic resolution from command-line arguments
    if (argc >= 4) {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    }

    vector<Triangle> h_triangles;
    if (!loadOBJ(argv[1], h_triangles)) {
        cerr << "Failed to load OBJ file!" << endl;
        return -1;
    }

    initOpenGL();

    GLFWwindow* window = glfwGetCurrentContext();

    while (!glfwWindowShouldClose(window)){ // Each iteration of this while loop represents rendering one frame
        updateCamera(window); // Process user input (keyboard for movement)
        renderHost(h_triangles);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); // Bind the PBO
        drawScreen();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind the PBO

        glfwSwapBuffers(window); // Displays the newly rendered frame
        glfwPollEvents(); // Processes user input (keyboard, mouse, etc.) and window events to receive close-window signals
    }

    glDeleteBuffers(1, &pbo); // Frees GPU memory used by the PBO
    cudaGraphicsUnregisterResource(cudaPBOResource); // Releases CUDA’s access to OpenGL’s buffer
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}