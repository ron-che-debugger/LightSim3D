#include "bvh.h"

/**
 * @file bvh.cpp
 * @brief Implements recursive construction of a bounding volume hierarchy (BVH) for triangle meshes.
 *
 * The BVH is built by partitioning triangles based on the spatial distribution of their centroids.
 * Each node in the hierarchy stores an axis-aligned bounding box (AABB) that encloses either a set
 * of primitives (leaf node) or two child nodes (internal node).
 *
 * The splitting axis is chosen based on the maximum extent of the centroid bounding box.
 * This structure accelerates ray-triangle intersection tests in ray tracing.
 */
namespace {

/**
 * @brief Recursively builds the bounding volume hierarchy (BVH) node for a given subset of triangles.
 *
 * This function partitions the triangle list and builds a binary tree of nodes, choosing the splitting axis
 * based on the extent of triangle centroids.
 *
 * @param triangles The list of all triangles in the scene.
 * @param indices The index list representing the current subset of triangles.
 * @param nodes The output BVH node array being constructed.
 * @param start The starting index into the indices vector for this subtree.
 * @param count The number of triangles in this subtree.
 * @param maxLeafSize Maximum number of triangles allowed in a leaf node.
 * @return The index of the created node in the nodes vector.
 */
int buildBVHRecursive(const vector<Triangle>& triangles,
                      vector<int>& indices,
                      vector<BVHNode>& nodes,
                      int start, int count,
                      int maxLeafSize) {
    BVHNode node;

    // Compute bounding box and centroid bounding box for current triangle range
    AABB bbox;
    bbox.min = make_float3( 1e30f,  1e30f,  1e30f);
    bbox.max = make_float3(-1e30f, -1e30f, -1e30f);
    AABB centroidBBox;
    centroidBBox.min = make_float3( 1e30f,  1e30f,  1e30f);
    centroidBBox.max = make_float3(-1e30f, -1e30f, -1e30f);
    
    for (int i = start; i < start + count; ++i) {
        int triIndex = indices[i];
        AABB triBox = computeTriangleAABB(triangles[triIndex]);
        bbox = unionAABB(bbox, triBox);

        float3 centroid = computeCentroid(triangles[triIndex]);
        AABB centBox;
        centBox.min = centroid;
        centBox.max = centroid;
        centroidBBox = unionAABB(centroidBBox, centBox);
    }

    int nodeIndex = nodes.size();
    nodes.push_back(node);

    // Base case: make a leaf node
    if (count <= maxLeafSize) {
        nodes[nodeIndex].bbox = bbox;
        nodes[nodeIndex].start = start;
        nodes[nodeIndex].count = count;
        nodes[nodeIndex].isLeaf = true;
        nodes[nodeIndex].left = -1;
        nodes[nodeIndex].right = -1;
    } else {
        // Choose axis with maximum extent
        float3 extent = MathUtils::float3_subtract(centroidBBox.max, centroidBBox.min);
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z)
            axis = 1;
        else if (extent.z > extent.x)
            axis = 2;

        // Sort indices based on centroids along the chosen axis
        sort(indices.begin() + start, indices.begin() + start + count, [&](int a, int b) {
            float3 ca = computeCentroid(triangles[a]);
            float3 cb = computeCentroid(triangles[b]);
            return (axis == 0 ? ca.x < cb.x : (axis == 1 ? ca.y < cb.y : ca.z < cb.z));
        });

        // Recursively build child nodes
        int mid = start + count / 2;
        int leftChild  = buildBVHRecursive(triangles, indices, nodes, start, count / 2, maxLeafSize);
        int rightChild = buildBVHRecursive(triangles, indices, nodes, mid, count - count / 2, maxLeafSize);

        nodes[nodeIndex].bbox = bbox;
        nodes[nodeIndex].isLeaf = false;
        nodes[nodeIndex].left = leftChild;
        nodes[nodeIndex].right = rightChild;
        nodes[nodeIndex].start = -1;
        nodes[nodeIndex].count = 0;
    }

    return nodeIndex;
}

} // anonymous namespace

/**
 * @brief Builds a bounding volume hierarchy (BVH) for the given list of triangles.
 *
 * The BVH is used to accelerate ray-scene intersection tests by organizing the triangles
 * in a tree structure based on their spatial locality.
 *
 * @param triangles The list of triangles to build the BVH for.
 * @param maxLeafSize The maximum number of triangles a leaf node can contain.
 * @return A BVH structure containing the built hierarchy and triangle index mapping.
 */
BVH buildBVH(const vector<Triangle> &triangles, int maxLeafSize) {
    BVH bvh;
    int numTriangles = triangles.size();
    bvh.triangleIndices.resize(numTriangles);
    for (int i = 0; i < numTriangles; i++) {
        bvh.triangleIndices[i] = i;
    }

    // Preallocate space for nodes to avoid reallocations
    bvh.nodes.reserve(numTriangles * 2);
    bvh.rootIndex = buildBVHRecursive(triangles, bvh.triangleIndices, bvh.nodes, 0, numTriangles, maxLeafSize);
    return bvh;
}