#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

#include "math_utils.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>

extern int width;
extern int height;

// Camera position & direction
extern float3 cameraPos;
extern float3 cameraDir;

// Mouse control variables
extern float yaw;
extern float pitch;
extern bool firstMouse;
extern float lastX;
extern float lastY;

// Camera Rotation
extern float objectYaw;
extern float objectPitch;

// OpenGL-CUDA interop resources
extern GLuint pbo;                            // Pixel buffer object
extern cudaGraphicsResource *cudaPBOResource; // CUDA-OpenGL interop resource

void updateCamera(GLFWwindow *window);

void updateMouse(GLFWwindow *window, double xpos, double ypos);

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);

void initOpenGL();

void drawScreen();
#endif