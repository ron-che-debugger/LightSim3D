#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "material.h"
#include "math_utils.h"
#include "tiny_obj_loader.h"
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Represents a triangle primitive with geometry, normal, material, and environment flag.
 */
struct Triangle {
    float3 v0, v1, v2;  /// Vertex positions of the triangle
    float3 normal;      /// Surface normal
    Material material;  /// Material properties for physically based rendering
    bool isEnvironment; /// Whether this triangle represents an environment light
};

/**
 * @brief Load triangle mesh data from an OBJ file and store as Triangle objects.
 *
 * @param filename: The path to the OBJ file to load.
 * @param triangles: Output vector to store loaded Triangle objects.
 * @return true if the file was successfully loaded, false otherwise.
 */
bool loadOBJ(const string &filename, vector<Triangle> &triangles);

/**
 * @brief Apply a named rendering effect to all triangles in the scene.
 *
 * Effects can include changes to material properties or visibility,
 * depending on the implementation.
 *
 * @param triangles: Vector of triangles to modify.
 * @param effect: The name of the effect to apply.
 */
void applyRenderingEffect(vector<Triangle> &triangles, const string &effect);

#endif // OBJ_LOADER_H