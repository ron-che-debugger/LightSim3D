#ifndef BVH_H
#define BVH_H

/**
 * @file bvh.h
 * @brief Defines data structures and utility functions for constructing a bounding volume hierarchy (BVH).
 *
 * This header provides:
 * - Axis-Aligned Bounding Box (AABB) definition and operations
 * - BVH node structure for acceleration of ray-primitive intersections
 * - Utility functions for bounding box computation and triangle centroid calculation
 * - Interface to construct a BVH from a triangle mesh
 */
#include "math_utils.h"
#include "obj_loader.h"
#include <algorithm>
#include <vector>

using namespace std;

// Axis-Aligned Bounding Box (AABB)
struct AABB {
    float3 min;
    float3 max;
};

// BVH Node definition
struct BVHNode {
    AABB bbox;
    int left;  /// Index of left child (if not a leaf)
    int right; /// Index of right child (if not a leaf)
    int start; /// Starting index in the triangle indices array (if leaf)
    int count; /// Number of triangles (if leaf)
    bool isLeaf;
};

/**
 * @brief Compute the union of two axis-aligned bounding boxes (AABB).
 *
 * @param a: First bounding box.
 * @param b: Second bounding box.
 * @return A new AABB that tightly contains both input boxes.
 */
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

/**
 * @brief Compute the axis-aligned bounding box (AABB) of a triangle.
 *
 * @param tri: The input triangle whose bounding box is to be computed.
 * @return An AABB that encloses the triangle.
 */
inline AABB computeTriangleAABB(const Triangle &tri) {
    AABB box;
    box.min = MathUtils::float3_min(MathUtils::float3_min(tri.v0, tri.v1), tri.v2);
    box.max = MathUtils::float3_max(MathUtils::float3_max(tri.v0, tri.v1), tri.v2);
    return box;
}

/**
 * @brief Compute the centroid (geometric center) of a triangle.
 *
 * @param tri: The input triangle.
 * @return A float3 representing the centroid of the triangle.
 */
inline float3 computeCentroid(const Triangle &tri) {
    float3 sum = MathUtils::float3_add(MathUtils::float3_add(tri.v0, tri.v1), tri.v2);
    return MathUtils::float3_scale(sum, 1.0f / 3.0f);
}

// BVH structure to hold the node array and the triangle index list.
struct BVH {
    vector<BVHNode> nodes;
    vector<int> triangleIndices;
    int rootIndex; /// Index of the root node in the nodes array
};

/**
 * @brief Construct a bounding volume hierarchy (BVH) from a list of triangles using recursive spatial partitioning.
 *
 * @param triangles: A vector of triangles used to construct the BVH.
 * @param maxLeafSize: The maximum number of triangles a leaf node can contain before splitting. Default is 4.
 * @return A BVH structure containing the hierarchy nodes and triangle index mapping.
 */
BVH buildBVH(const vector<Triangle> &triangles, int maxLeafSize = 4);

#endif // BVH_H