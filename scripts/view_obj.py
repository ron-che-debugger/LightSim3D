import argparse
import trimesh
import matplotlib.pyplot as plt

def load_obj(file_path):
    """ Load an OBJ file and extract vertices and faces """
    mesh = trimesh.load_mesh(file_path)
    return mesh.vertices, mesh.faces

def plot_mesh(vertices, faces):
    """ Plot a 3D mesh using matplotlib """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(*zip(*vertices), triangles=faces, color='cyan', edgecolor='black')
    plt.show()

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize an OBJ file")
    parser.add_argument("file_path", type=str, help="Path to the OBJ file")
    
    args = parser.parse_args()

    # Load and visualize the OBJ file
    vertices, faces = load_obj(args.file_path)
    plot_mesh(vertices, faces)