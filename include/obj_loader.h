#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "material.h"
#include "math_utils.h"
#include "tiny_obj_loader.h"
#include <iostream>
#include <vector>

using namespace std;

struct Triangle {
    float3 v0, v1, v2;
    float3 normal;
    Material material; // Material properties for PBR shading
    bool isEnvironment;
};

bool loadOBJ(const string &filename, vector<Triangle> &triangles);

void applyRenderingEffect(vector<Triangle> &triangles, const string &effect);

#endif // OBJ_LOADER_H
