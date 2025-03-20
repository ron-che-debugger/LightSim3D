#ifndef BVH_H
#define BVH_H

#include "math_utils.h"
#include "obj_loader.h"
#include <vector>
#include <algorithm>

using namespace std;

// Axis-Aligned Bounding Box (AABB)
struct AABB {
    float3 min;
    float3 max;
};

// BVH Node definition
struct BVHNode {
    AABB bbox;
    int left;   // Index of left child (if not a leaf)
    int right;  // Index of right child (if not a leaf)
    int start;  // Starting index in the triangle indices array (if leaf)
    int count;  // Number of triangles (if leaf)
    bool isLeaf;
};

// Utility: compute the union of two AABBs
inline AABB unionAABB(const AABB &a, const AABB &b) {
    AABB ret;
    ret.min = make_float3(fminf(a.min.x, b.min.x),
                          fminf(a.min.y, b.min.y),
                          fminf(a.min.z, b.min.z));
    ret.max = make_float3(fmaxf(a.max.x, b.max.x),
                          fmaxf(a.max.y, b.max.y),
                          fmaxf(a.max.z, b.max.z));
    return ret;
}

// Compute AABB for a triangle
inline AABB computeTriangleAABB(const Triangle &tri) {
    AABB box;
    box.min = MathUtils::float3_min(MathUtils::float3_min(tri.v0, tri.v1), tri.v2);
    box.max = MathUtils::float3_max(MathUtils::float3_max(tri.v0, tri.v1), tri.v2);
    return box;
}

// Compute centroid of a triangle
inline float3 computeCentroid(const Triangle &tri) {
    float3 sum = MathUtils::float3_add(MathUtils::float3_add(tri.v0, tri.v1), tri.v2);
    return MathUtils::float3_scale(sum, 1.0f / 3.0f);
}

// BVH structure to hold the node array and the triangle index list.
struct BVH {
    vector<BVHNode> nodes;
    vector<int> triangleIndices;
    int rootIndex; // index of the root node in nodes array
};

// Build BVH from a list of triangles; maxLeafSize determines when to stop splitting.
BVH buildBVH(const vector<Triangle> &triangles, int maxLeafSize = 4);

#endif // BVH_H