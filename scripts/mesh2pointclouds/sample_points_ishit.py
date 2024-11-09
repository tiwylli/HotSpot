import open3d as o3d
import numpy as np
import os
import glob

def fuse_pcd(input_path, output_path, filename, number_of_points=10**6):
    """Converts a mesh file to a point cloud and saves it."""
    # Read mesh file (supports both .obj and .ply)
    mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=True)

    if not mesh.has_vertices():
        print(f"Warning: {filename} does not have valid vertices. Skipping.")
        return

    # Transform mesh vertices (Blender to OpenCV coordinate system)
    blender_to_opencv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # mesh_vertices = np.asarray(mesh.vertices) @ blender_to_opencv.T
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    # Sample points from the mesh
    sampled_pcd = mesh.sample_points_uniformly(number_of_points)

    # Save the point cloud with the same filename in the output path
    output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.ply")
    o3d.io.write_point_cloud(output_file, sampled_pcd, compressed=True)
    print(f"Saved point cloud: {output_file}")

if __name__ == "__main__":
    input_dir = "/pv-stao3/SPIN/data/NIE/ground_truth"  # Input directory
    output_dir = "/pv-stao3/SPIN/data/NIE/point_cloud"  # Output directory

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all .obj and .ply files in the input directory
    mesh_files = glob.glob(os.path.join(input_dir, "*.obj")) + glob.glob(os.path.join(input_dir, "*.ply"))

    for mesh_file in mesh_files:
        filename = os.path.basename(mesh_file)
        print(f"Processing: {filename}")
        fuse_pcd(mesh_file, output_dir, filename, number_of_points=100000)
