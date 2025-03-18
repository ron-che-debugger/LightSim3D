#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "tiny_obj_loader.h"
#include "math_utils.h"
#include <iostream>
#include <vector>

using namespace std;

struct Triangle {
    float3 v0, v1, v2;
    float3 normal;
};

bool loadOBJ(const string &filename, vector<Triangle> &triangles);

#endif