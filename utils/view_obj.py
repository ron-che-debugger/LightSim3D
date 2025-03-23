#!/usr/bin/env python3
"""
plot_obj.py

Load & visualize an OBJ file that may contain triangles or quads.
If a face has 4 vertices, it is split into two triangles.

Usage:
    python plot_obj.py my_model.obj
"""

import argparse
import matplotlib.pyplot as plt

def load_obj(path):
    """ Load an OBJ file, supporting triangular or quad faces. """
    vertices = []
    triangles = []
    with open(path, 'r') as f:
        for line in f:
            # Remove inline comments and whitespace
            line = line.split('#', 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == 'v':
                # OBJ vertex line -> store as float [x, y, z]
                vertices.append(list(map(float, parts[1:])))
            elif parts[0] == 'f':
                # Convert OBJ's 1-based indices to 0-based
                face_indices = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]

                # Triangular face
                if len(face_indices) == 3:
                    triangles.append(face_indices)
                # Quad face -> split into two triangles
                elif len(face_indices) == 4:
                    a, b, c, d = face_indices
                    triangles.append([a, b, c])
                    triangles.append([a, c, d])
                else:
                    # For faces with more than 4 vertices, you'd need a more general polygon triangulation.
                    # Here we simply skip them or raise an error:
                    print(f"Warning: skipping face with {len(face_indices)} vertices.")
                    continue
    return vertices, triangles

def plot_mesh(vertices, faces):
    """
    Plot a 3D mesh using matplotlib's plot_trisurf.
    'faces' must be a list of triangle indices (3-element lists).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # *zip(*vertices) unpacks the vertex list into x, y, z for plotting
    ax.plot_trisurf(*zip(*vertices), triangles=faces, color='cyan', edgecolor='black')
    ax.set_box_aspect((1, 1, 1))
    plt.title("OBJ Mesh Preview")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load & visualize an OBJ file — supports triangles and quads."
    )
    parser.add_argument("file_path", help="Path to the OBJ file")
    args = parser.parse_args()

    try:
        verts, faces = load_obj(args.file_path)
        plot_mesh(verts, faces)
    except Exception as e:
        print(f"Failed to load '{args.file_path}': {e}")