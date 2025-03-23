/**
 * @file opengl_utils.cpp
 * @brief Implements OpenGL window setup, user input handling, and CUDA-OpenGL interoperability.
 *
 * This module handles:
 * - OpenGL context setup with GLFW and GLEW
 * - Camera and object control via keyboard and mouse input
 * - CUDA-OpenGL interop using pixel buffer objects (PBO)
 * - Input callbacks for real-time interaction
 *
 * Enables real-time interactive rendering and camera movement in the ray tracer.
 */
#include "opengl_utils.h"

using namespace std;

extern int width;
extern int height;

float3 cameraPos = make_float3(0, 0, -5);
float3 cameraDir = make_float3(0, 0, 1);

float yaw = 0.0f;
float pitch = 0.0f;
bool firstMouse = true;
float lastX = width / 2.0f;
float lastY = height / 2.0f;

float objectYaw = 0.0f;
float objectPitch = 0.0f;

bool cursorLocked = true;  // Track whether the cursor is locked

GLuint pbo;
cudaGraphicsResource* cudaPBOResource;

/**
 * @brief Update camera position using WASD and vertical keys based on user input.
 *
 * @param window Pointer to the GLFW window handling input events.
 */
void updateCamera(GLFWwindow* window) {
    float speed = 0.1f;
    float3 right = MathUtils::normalize(MathUtils::cross(make_float3(0, 1, 0), cameraDir));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos = MathUtils::float3_add(cameraPos, MathUtils::float3_scale(cameraDir, speed));
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos = MathUtils::float3_subtract(cameraPos, MathUtils::float3_scale(cameraDir, speed));
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos = MathUtils::float3_subtract(cameraPos, MathUtils::float3_scale(right, speed));
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos = MathUtils::float3_add(cameraPos, MathUtils::float3_scale(right, speed));
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos.y += speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos.y -= speed;
}

/**
 * @brief Update camera rotation based on mouse movement.
 *
 * This function is triggered by the mouse callback and updates pitch and yaw angles.
 *
 * @param window Pointer to the GLFW window.
 * @param xpos Current X position of the mouse.
 * @param ypos Current Y position of the mouse.
 */
void updateMouse(GLFWwindow* window, double xpos, double ypos) {
    // Only update rotation if the cursor is locked
    if (!cursorLocked)
        return;

    if (firstMouse) {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos - lastX);
    float yoffset = static_cast<float>(lastY - ypos);
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);

    float sensitivity = 0.08f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    objectYaw += xoffset * (M_PI / 180.0f);
    objectPitch += yoffset * (M_PI / 180.0f);

    if (objectPitch > 89.0f * (M_PI / 180.0f))
        objectPitch = 89.0f * (M_PI / 180.0f);
    if (objectPitch < -89.0f * (M_PI / 180.0f))
        objectPitch = -89.0f * (M_PI / 180.0f);
}

/**
 * @brief GLFW key callback for ESC key to unlock the cursor.
 *
 * @param window Pointer to the GLFW window.
 * @param key Key code of the pressed key.
 * @param scancode Platform-specific scan code.
 * @param action Action type (press, release).
 * @param mods Modifier keys (shift, alt, etc.).
 */
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  // Unlock the cursor
        cursorLocked = false; // Stop mouse rotation when cursor is unlocked
    }
}

/**
 * @brief GLFW mouse button callback to re-lock the cursor on left click.
 *
 * @param window Pointer to the GLFW window.
 * @param button Mouse button code.
 * @param action Action type (press or release).
 * @param mods Modifier keys.
 */
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Re-lock the cursor
        cursorLocked = true; // Resume mouse rotation when cursor is locked
        firstMouse = true;   // Reset firstMouse to avoid jump on re-locking
    }
}

/**
 * @brief Initialize GLFW, GLEW, input callbacks, and set up CUDA-OpenGL interop with a pixel buffer object.
 */
void initOpenGL(){
    // Initialize GLFW
    if (!glfwInit()){
        cerr << "Failed to initialize GLFW" << endl;
        exit(EXIT_FAILURE);
    }

    // Create OpenGL Window
    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Raytracer", NULL, NULL);
    if (!window){
        cerr << "Failed to create OpenGL window" << endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window); // Tell OpenGL which window it should render into
    glfwSetCursorPosCallback(window, updateMouse);
    glfwSetKeyCallback(window, keyCallback);       // Register ESC key handling
    glfwSetMouseButtonCallback(window, mouseButtonCallback);  // Register mouse click handling
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Initially disable cursor

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        exit(EXIT_FAILURE);
    }

    // Generate and bind PBO
    glGenBuffers(1, &pbo);  // Create 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); // Make pbo the active Pixel Buffer Object for OpenGL
    // Allocates memory for the buffer but does NOT fill it yet
    // GL_DYNAMIC_DRAW - Usage hint: Data will be modified frequently
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbinds the current PBO, preventing accidental modifications

    // Register PBO with CUDA
    // cudaGraphicsRegisterFlagsWriteDiscard - tells CUDA that it does NOT need to preserve previous data when mapping the buffer
    cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

/**
 * @brief Display the CUDA-rendered image by drawing the contents of the PBO.
 */
void drawScreen() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}