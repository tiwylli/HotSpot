import open3d as o3d
import numpy as np
import os
import glob


def fuse_pcd(input_path, output_path, filename, number_of_points):
    """Converts a mesh file to a point cloud and saves it."""
    print(f"Loading mesh from: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=True)

    # Check if the mesh is valid
    if not mesh.has_vertices():
        print(f"Warning: {filename} does not have valid vertices. Skipping.")
        return

    if not mesh.has_triangles():
        print(f"Warning: {filename} does not have valid faces. Skipping.")
        return

    # Print basic mesh information
    print(
        f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles."
    )

    # Sample points from the mesh
    try:
        sampled_pcd = mesh.sample_points_uniformly(number_of_points)
    except Exception as e:
        print(f"Error during sampling points for {filename}: {e}")
        return

    # Save the point cloud with the same filename in the output path
    output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.ply")
    try:
        o3d.io.write_point_cloud(output_file, sampled_pcd, compressed=True)
        print(f"Saved point cloud: {output_file}")
    except Exception as e:
        print(f"Error saving point cloud for {filename}: {e}")


if __name__ == "__main__":
    input_dir = "data/complex_vsphere/ground_truth"  # Input directory
    output_dir = "data/complex_vsphere/point_cloud"  # Output directory

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all .obj and .ply files in the input directory
    mesh_files = glob.glob(os.path.join(input_dir, "*.obj")) + glob.glob(
        os.path.join(input_dir, "*.ply")
    )

    if not mesh_files:
        print(f"No .obj or .ply files found in directory: {input_dir}")
    else:
        for mesh_file in mesh_files:
            filename = os.path.basename(mesh_file)
            print(f"Processing: {filename}")

            # Set point count based on filename
            if "double" in filename.lower():
                points = 200000
            elif "triple" in filename.lower():
                points = 300000
            else:
                points = 100000  # Default for other cases

            fuse_pcd(mesh_file, output_dir, filename, number_of_points=points)
