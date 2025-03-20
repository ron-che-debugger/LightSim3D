#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "math_utils.h"
#include "tiny_obj_loader.h"
#include <iostream>
#include <vector>

using namespace std;

struct Triangle {
    float3 v0, v1, v2;
    float3 normal;
    float3 emission = make_float3(0.0f, 0.0f, 0.0f); // Default: No emission
};

bool loadOBJ(const string &filename, vector<Triangle> &triangles);

#endif