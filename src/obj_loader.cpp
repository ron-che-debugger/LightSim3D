#define TINYOBJLOADER_IMPLEMENTATION
#include "obj_loader.h"

using namespace std;

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
            
            // Set default material properties:
            tri.material.albedo   = make_float3(0.8f, 0.8f, 0.8f); // Diffuse color
            tri.material.metallic = 0.0f;                        // Default: non-metallic
            tri.material.roughness= 0.5f;                        // Moderate roughness

            // If the face is upward–facing or right-facing, treat it as a light source.
            if (tri.normal.y > 0.9f) {
                tri.material.emission = make_float3(5.0f, 5.0f, 5.0f);
            } else {
                tri.material.emission = make_float3(0.0f, 0.0f, 0.0f);
            }
            
            triangles.push_back(tri);
        }
    }

    return true;
}