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

GLuint pbo;
cudaGraphicsResource* cudaPBOResource;

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

    objectYaw += xoffset * (M_PI / 180.0f);
    objectPitch += yoffset * (M_PI / 180.0f);

    if (objectPitch > 89.0f * (M_PI / 180.0f))
        objectPitch = 89.0f * (M_PI / 180.0f);
    if (objectPitch < -89.0f * (M_PI / 180.0f))
        objectPitch = -89.0f * (M_PI / 180.0f);
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