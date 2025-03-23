import argparse
import trimesh
import matplotlib.pyplot as plt
import io

def load_obj(path):
    verts = []
    faces = []
    with open(path, 'r') as f:
        for line in f:
            # Remove inline comments and whitespace
            line = line.split('#', 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == 'v':
                verts.append(list(map(float, parts[1:])))
            elif parts[0] == 'f':
                # OBJ faces are 1‑indexed — convert to 0‑indexed
                faces.append([int(idx.split('/')[0]) - 1 for idx in parts[1:]])
    return verts, faces

def plot_mesh(vertices, faces):
    """ Plot a 3D mesh using matplotlib """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(*zip(*vertices), triangles=faces, color='cyan', edgecolor='black')
    ax.set_box_aspect((1,1,1))
    plt.title("OBJ Mesh Preview")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load & visualize an OBJ file — safely ignores '#' comment lines"
    )
    parser.add_argument("file_path", help="Path to the OBJ file")
    args = parser.parse_args()

    try:
        verts, faces = load_obj(args.file_path)
        plot_mesh(verts, faces)
    except Exception as e:
        print(f"Failed to load '{args.file_path}': {e}")
