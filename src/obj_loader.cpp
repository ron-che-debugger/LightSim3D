#define TINYOBJLOADER_IMPLEMENTATION

/**
 * @file obj_loader.cpp
 * @brief Implements mesh loading from OBJ files and material effect application for rendering.
 *
 * Uses TinyObjLoader to parse geometry data and converts it into a list of triangles with
 * computed normals and default PBR material settings. Also supports basic material overrides
 * for visual debugging or styling (e.g., matte, metal).
 */
#include "obj_loader.h"

using namespace std;

/**
 * @brief Load geometry from an OBJ file and convert it into a list of triangles.
 *
 * This function uses TinyObjLoader to parse an OBJ file and creates a triangle list suitable
 * for ray tracing. Vertex positions are extracted, face normals are computed, and default
 * material properties are applied to each triangle.
 *
 * @param filename Path to the OBJ file to be loaded.
 * @param triangles Output list where the loaded triangles will be stored.
 * @return True if the OBJ file is successfully loaded, false otherwise.
 */
bool loadOBJ(const string& filename, vector<Triangle>& triangles){
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())){
        cerr << "Error loading OBJ: " << err << endl;
        return false;
    }

    // Convert OBJ faces into triangles
    for (const auto& shape : shapes){
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3){
            Triangle tri;

            // Get indices for the triangle vertices.
            int idx0 = shape.mesh.indices[i + 0].vertex_index;
            int idx1 = shape.mesh.indices[i + 1].vertex_index;
            int idx2 = shape.mesh.indices[i + 2].vertex_index;

            tri.v0 = make_float3(
                attrib.vertices[3 * idx0 + 0],
                attrib.vertices[3 * idx0 + 1],
                attrib.vertices[3 * idx0 + 2]
            );

            tri.v1 = make_float3(
                attrib.vertices[3 * idx1 + 0],
                attrib.vertices[3 * idx1 + 1],
                attrib.vertices[3 * idx1 + 2]
            );

            tri.v2 = make_float3(
                attrib.vertices[3 * idx2 + 0],
                attrib.vertices[3 * idx2 + 1],
                attrib.vertices[3 * idx2 + 2]
            );

            // Compute the triangle’s face normal (assumes counterclockwise winding).
            float3 edge1 = MathUtils::float3_subtract(tri.v1, tri.v0);
            float3 edge2 = MathUtils::float3_subtract(tri.v2, tri.v0);
            tri.normal = MathUtils::normalize(MathUtils::cross(edge1, edge2));

            // Set default material properties
            tri.material.albedo   = make_float3(1.0f, 1.0f, 1.0f);
            tri.material.metallic = 0.0f;
            tri.material.emission = make_float3(0.0f, 0.0f, 0.0f);
            tri.isEnvironment = false;

            triangles.push_back(tri);
        }
    }

    return true;
}

/**
 * @brief Apply a rendering effect to object geometry based on the selected material style.
 *
 * This function modifies the material properties of non-emissive, non-environment triangles.
 * Supported effects:
 * - "metal": sets metallic = 1.0
 * - "matte": sets metallic = 0.0
 * - "default": sets metallic = 0.1
 *
 * @param triangles The triangle list to update.
 * @param effect The name of the rendering effect to apply.
 */
void applyRenderingEffect(vector<Triangle>& triangles, const string &effect) {
    for (auto &tri : triangles) {
        // Only update non-environment and non-emissive (object) triangles.
        if (!tri.isEnvironment &&
            tri.material.emission.x == 0.0f &&
            tri.material.emission.y == 0.0f &&
            tri.material.emission.z == 0.0f) {

            if (effect == "metal") {
                tri.material.metallic = 1.0f;   // Fully metallic
            } else if (effect == "matte") {
                tri.material.metallic = 0.0f;   // Purely diffuse
            } else if (effect == "default") {
                tri.material.metallic = 0.7f;
            }
        }
    }
}