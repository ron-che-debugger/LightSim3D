#ifndef OPENGL_UTILS_H
#define OPENGL_UTILS_H

/**
 * @file opengl_utils.h
 * @brief Utility functions and global state for OpenGL windowing, input handling, and CUDA interop.
 *
 * Provides:
 * - OpenGL context setup with GLFW and GLEW
 * - Camera and object control via keyboard and mouse input
 * - CUDA-OpenGL interop using pixel buffer objects (PBO)
 * - Input callbacks for real-time interaction
 *
 * Used to display CUDA-rendered frames and enable interactive navigation.
 */
#include "math_utils.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>

// Window dimensions
extern int width;
extern int height;

// Camera position and viewing direction
extern float3 cameraPos; /// Position of the camera in world space
extern float3 cameraDir; /// Forward direction vector of the camera

// Mouse input tracking
extern float yaw;       /// Horizontal angle of camera rotation
extern float pitch;     /// Vertical angle of camera rotation
extern bool firstMouse; /// Flag for initializing mouse input
extern float lastX;     /// Last recorded X position of mouse
extern float lastY;     /// Last recorded Y position of mouse

// Object orientation
extern float objectYaw;   /// Yaw angle for object-space rotation
extern float objectPitch; /// Pitch angle for object-space rotation

// OpenGL-CUDA interop resources
extern GLuint pbo;                            /// Pixel Buffer Object used for rendering
extern cudaGraphicsResource *cudaPBOResource; /// CUDA registration of the pixel buffer

/**
 * @brief Update the camera direction and position based on user input.
 *
 * @param window Pointer to the GLFW window handling input events.
 */
void updateCamera(GLFWwindow *window);

/**
 * @brief Update camera rotation based on mouse movement.
 *
 * @param window Pointer to the GLFW window.
 * @param xpos Current X position of the mouse.
 * @param ypos Current Y position of the mouse.
 */
void updateMouse(GLFWwindow *window, double xpos, double ypos);

/**
 * @brief Handle keyboard events.
 *
 * @param window Pointer to the GLFW window.
 * @param key Key code of the pressed key.
 * @param scancode Platform-specific key identifier.
 * @param action Action (press, release, repeat).
 * @param mods Modifier keys held during the event.
 */
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

/**
 * @brief Handle mouse button input events.
 *
 * @param window Pointer to the GLFW window.
 * @param button Mouse button pressed or released.
 * @param action Action (press or release).
 * @param mods Modifier keys held during the event.
 */
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);

/**
 * @brief Initialize OpenGL context, GLEW, and create the render window.
 */
void initOpenGL();

/**
 * @brief Draw the contents of the screen from the pixel buffer object.
 */
void drawScreen();

#endif