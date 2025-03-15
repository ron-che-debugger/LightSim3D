#include "obj_loader.h"
#include "raytracer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

using namespace std;

int width = 800;   // Define the image width
int height = 600;  // Define the image height

void saveImage(const char* filename, uchar4* pixels, int width, int height){
    FILE* f = fopen(filename, "wb"); // Open file in binary write mode

    fprintf(f, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        fputc(pixels[i].x, f);  // Red
        fputc(pixels[i].y, f);  // Green
        fputc(pixels[i].z, f);  // Blue
    }

    fclose(f);
    cout << "Image saved as " << filename << endl;
}

void renderHost(const vector<Triangle>& h_triangles, uchar4* h_pixels){
    Triangle* d_triangles;  
    cudaError_t err;

    err = cudaMalloc(&d_triangles, h_triangles.size() * sizeof(Triangle));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_triangles: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(d_triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_triangles: %s\n", cudaGetErrorString(err));
        return;
    }

    uchar4* d_pixels;
    err = cudaMalloc(&d_pixels, width * height * sizeof(uchar4));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_pixels: %s\n", cudaGetErrorString(err));
        return;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    renderKernel<<<gridSize, blockSize>>>(d_pixels, width, height, d_triangles, h_triangles.size());
    err = cudaGetLastError();  // Check for launch errors
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize(); // Ensure kernel execution is completed before copying results

    err = cudaMemcpy(h_pixels, d_pixels, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for h_pixels: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaFree(d_triangles);
    cudaFree(d_pixels);
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

    uchar4* h_pixels = new uchar4[width * height]; 
    renderHost(h_triangles, h_pixels);

    // Save the output as an image
    saveImage("output.ppm", h_pixels, width, height);

    delete[] h_pixels;
    return 0;
}
