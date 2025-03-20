#include "bvh.h"

namespace {

// Recursive BVH builder helper. It builds a node for triangles in indices[start, start+count)
// and returns its index in the nodes vector.
int buildBVHRecursive(const vector<Triangle>& triangles,
                      vector<int>& indices,
                      vector<BVHNode>& nodes,
                      int start, int count,
                      int maxLeafSize) {
    BVHNode node;
    // Compute bounding box for all triangles in this node.
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
    
    // If the number of triangles is small, make this node a leaf.
    if (count <= maxLeafSize) {
        nodes[nodeIndex].bbox = bbox;
        nodes[nodeIndex].start = start;
        nodes[nodeIndex].count = count;
        nodes[nodeIndex].isLeaf = true;
        nodes[nodeIndex].left = -1;
        nodes[nodeIndex].right = -1;
    } else {
        // Choose the axis along which the centroids have the greatest extent.
        float3 extent = MathUtils::float3_subtract(centroidBBox.max, centroidBBox.min);
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z)
            axis = 1;
        else if (extent.z > extent.x)
            axis = 2;
        
        // Sort triangle indices based on centroid along chosen axis.
        sort(indices.begin() + start, indices.begin() + start + count, [&](int a, int b) {
            float3 ca = computeCentroid(triangles[a]);
            float3 cb = computeCentroid(triangles[b]);
            return (axis == 0 ? ca.x < cb.x : (axis == 1 ? ca.y < cb.y : ca.z < cb.z));
        });
        
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

BVH buildBVH(const vector<Triangle> &triangles, int maxLeafSize) {
    BVH bvh;
    int numTriangles = triangles.size();
    bvh.triangleIndices.resize(numTriangles);
    for (int i = 0; i < numTriangles; i++) {
        bvh.triangleIndices[i] = i;
    }
    // Reserve an upper bound for nodes (2*numTriangles is safe for binary trees).
    bvh.nodes.reserve(numTriangles * 2);
    bvh.rootIndex = buildBVHRecursive(triangles, bvh.triangleIndices, bvh.nodes, 0, numTriangles, maxLeafSize);
    return bvh;
}