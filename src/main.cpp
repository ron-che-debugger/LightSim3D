#include "obj_loader.h"
#include "raytracer.h"
#include "mathutils.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

using namespace std;

#define M_PI 3.14159265358979323846

// Window dimensions
int width = 800;   // Define the image width (default)
int height = 600;  // Define the image height (default)

// Object Rotation
float objectYaw = 0.0f;
float objectPitch = 0.0f;

// OpenGL-CUDA interop resources
GLuint pbo; // Pixel buffer object
cudaGraphicsResource* cudaPBOResource;  // CUDA-OpenGL interop resource

// Camera controls
float3 cameraPos = make_float3(0, 0, -5);
float3 cameraDir = make_float3(0, 0, 1);

// Mouse rotation controls
float yaw = 0.0f;
float pitch = 0.0f;
bool firstMouse = true;
float lastX = width / 2.0f;
float lastY = height / 2.0f;

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

void updateMouse(GLFWwindow* window, double xpos, double ypos) {
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

    objectYaw += xoffset * (static_cast<float>(M_PI) / 180.0f);
    objectPitch += yoffset * (static_cast<float>(M_PI) / 180.0f);

    if (objectPitch > 89.0f * (static_cast<float>(M_PI) / 180.0f))
        objectPitch = 89.0f * (static_cast<float>(M_PI) / 180.0f);
    if (objectPitch < -89.0f * (static_cast<float>(M_PI) / 180.0f))
        objectPitch = -89.0f * (static_cast<float>(M_PI) / 180.0f);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  // Unlock the cursor
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Re-lock the cursor
    }
}

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

void renderHost(const vector<Triangle>& h_triangles){
    Triangle* d_triangles;  
    cudaMalloc(&d_triangles, h_triangles.size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, h_triangles.data(), h_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    uchar4* d_pixels;
    size_t numBytes;
    // Locks the OpenGL PBO for CUDA access (prevents OpenGL from using it)
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    // Retrieves a CUDA device pointer (d_pixels) to the PBO's memory 
    cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &numBytes, cudaPBOResource); 

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    renderKernel<<<gridSize, blockSize>>>(d_pixels, width, height, d_triangles, 
        h_triangles.size(), cameraPos, cameraDir, 
        objectYaw, objectPitch);

    // Unlocks the PBO so OpenGL can use it again
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);

    cudaFree(d_triangles);
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
