#!/usr/bin/env python3
"""
fix_obj_normals.py

Improved script that loads an OBJ file, computes an approximate object centroid,
and for every face (triangle, quad, or n-gon) does a simple "fan triangulation"
to compute a representative face normal. If the face normal is inward
(dot product with (faceCentroid - objectCentroid) < 0), it flips the face.

Usage:
    python fix_obj_normals.py input.obj output.obj
"""

import argparse
import numpy as np

def load_obj(filename):
    """
    Loads an OBJ file and returns a list of vertices plus a dictionary of object/group faces.
    Each 'object' or 'group' is keyed by its name; the value is a list of raw face lines.
    """
    vertices = []
    # Use "default" if no object/group is defined.
    objects = {"default": []}
    current_obj = "default"
    header = []

    with open(filename, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Check if this is a vertex line
            if stripped.startswith("v "):
                parts = stripped.split()
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                vertices.append(vertex)
                header.append(line)  # Keep original vertex line in header
            # Check if this is an object or group line
            elif stripped.startswith("o ") or stripped.startswith("g "):
                current_obj = stripped
                objects[current_obj] = []
                header.append(line)
            # Check if this is a face line
            elif stripped.startswith("f "):
                objects[current_obj].append(line)
            else:
                # Keep any other lines in the header
                header.append(line)

    return vertices, objects, header


def parse_face_line(face_line):
    """
    Parses a face line from an OBJ file and returns a list of vertex indices (0-indexed).
    E.g. "f 1/2/3 4/5/6" -> [0, 3] ignoring texture/normal indices.
    """
    parts = face_line.split()
    if parts[0] != "f":
        return []
    indices = []
    for part in parts[1:]:
        # OBJ is 1-based; we convert to 0-based
        idx_str = part.split('/')[0]
        idx = int(idx_str) - 1
        indices.append(idx)
    return indices


def construct_face_line(face_line):
    """
    Reverses the winding order of the face by reversing the vertex tokens.
    Preserves any texture/normal data. E.g. "f v1/t1/n1 v2/t2/n2 ..." -> "f vN/tN/nN ... v1/t1/n1".
    """
    tokens = face_line.split()
    if tokens[0] != "f":
        return face_line
    reversed_tokens = tokens[1:][::-1]
    new_line = "f " + " ".join(reversed_tokens) + "\n"
    return new_line


def compute_face_normal_and_centroid(vertices, face_indices):
    """
    Computes a representative normal and centroid for a face (triangle, quad, or n-gon).
    - face_indices: list of vertex indices for the face
    - Uses "fan triangulation" from the first vertex. Summation of cross products of each triangle.
    - Returns (normal, centroid), or (None, None) if degenerate.
    """
    if len(face_indices) < 3:
        return None, None

    # Gather face vertices
    face_verts = [vertices[i] for i in face_indices]

    # Face centroid is the average of all its vertices
    face_centroid = np.mean(face_verts, axis=0)

    # "Fan triangulation" from face_verts[0]
    v0 = face_verts[0]
    normal_sum = np.array([0.0, 0.0, 0.0], dtype=float)

    for i in range(1, len(face_verts) - 1):
        v1 = face_verts[i]
        v2 = face_verts[i + 1]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal_sum += np.cross(edge1, edge2)

    norm_len = np.linalg.norm(normal_sum)
    if norm_len < 1e-12:
        # Degenerate face
        return None, None
    normal = normal_sum / norm_len

    return normal, face_centroid


def fix_normals(vertices, objects):
    """
    For each object, compute the centroid of all used vertices. Then for each face:
    - parse the face's vertex indices
    - compute the face's normal and centroid
    - if dot(normal, faceCentroid - objectCentroid) < 0, flip the face
    Returns a new dictionary with updated face lines.
    """
    fixed_objects = {}

    for obj_name, face_lines in objects.items():
        # Collect all vertex indices used by this object
        used_indices = set()
        for face_line in face_lines:
            inds = parse_face_line(face_line)
            used_indices.update(inds)

        if used_indices:
            obj_vertices = np.array([vertices[i] for i in used_indices])
            object_centroid = np.mean(obj_vertices, axis=0)
        else:
            object_centroid = np.array([0.0, 0.0, 0.0])

        fixed_faces = []

        for face_line in face_lines:
            face_indices = parse_face_line(face_line)
            if len(face_indices) < 3:
                # Not a valid polygon, skip
                fixed_faces.append(face_line)
                continue

            normal, face_centroid = compute_face_normal_and_centroid(vertices, face_indices)
            if normal is None:
                # Degenerate face, keep as-is
                fixed_faces.append(face_line)
                continue

            # Vector from object centroid to face centroid
            to_face = face_centroid - object_centroid

            # If dot < 0, the face is "inward," so flip
            if np.dot(normal, to_face) < 0:
                fixed_faces.append(construct_face_line(face_line))
            else:
                fixed_faces.append(face_line)

        fixed_objects[obj_name] = fixed_faces

    return fixed_objects


def write_obj(filename, vertices, fixed_objects, header):
    """
    Writes a new OBJ file:
      - All vertex lines first
      - Then each object's face lines
    The 'header' lines are not re‐written in the same order,
    but if you want them, you can also include them here if desired.
    """
    with open(filename, 'w') as f:
        # Re‐write all vertex lines
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write each object's lines
        for obj_name, face_lines in fixed_objects.items():
            if obj_name != "default":
                f.write(obj_name + "\n")
            for line in face_lines:
                f.write(line)

    print("Fixed OBJ written to:", filename)


def main():
    parser = argparse.ArgumentParser(description="Fix face normals in an OBJ file so every face is outward-facing.")
    parser.add_argument("input", help="Input OBJ file")
    parser.add_argument("output", help="Output OBJ file")
    args = parser.parse_args()

    vertices, objects, header = load_obj(args.input)
    fixed_objects = fix_normals(vertices, objects)
    write_obj(args.output, vertices, fixed_objects, header)


if __name__ == "__main__":
    main()