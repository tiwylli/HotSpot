import open3d as o3d
import numpy as np
import os


def find_centroid(mesh):
    """Calculate the centroid of the mesh."""
    vertices = np.asarray(mesh.vertices)
    centroid = np.mean(vertices, axis=0)
    return centroid


def transform_mesh(mesh, centroid, scale=1.0, rotation_angle=0.0):
    """Apply scaling and rotation transformations to the mesh."""
    # Scale the mesh
    vertices = np.asarray(mesh.vertices) - centroid  # Center at the origin
    vertices *= scale  # Apply scaling
    vertices += centroid  # Translate back

    # Rotate the mesh
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [0, 0, np.radians(rotation_angle)]
    )
    vertices = np.dot(vertices - centroid, rotation_matrix.T) + centroid

    # Update the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def copy_mesh(mesh):
    """Manually copy a mesh object."""
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
    new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles).copy())
    new_mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertex_normals).copy()
    )
    new_mesh.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh.triangle_normals).copy()
    )
    return new_mesh


def create_layered_spheres(input_path, output_folder, scales, rotations):
    """Create double and triple-layered spheres."""
    mesh = o3d.io.read_triangle_mesh(input_path)
    if not mesh.has_vertices():
        print("Mesh does not have valid vertices. Exiting.")
        return

    centroid = find_centroid(mesh)

    # Original mesh
    meshes = [mesh]

    # Create transformed layers
    for scale, rotation in zip(scales, rotations):
        transformed_mesh = copy_mesh(mesh)
        transformed_mesh = transform_mesh(
            transformed_mesh, centroid, scale=scale, rotation_angle=rotation
        )
        meshes.append(transformed_mesh)

    # Combine layers
    combined_mesh = o3d.geometry.TriangleMesh()
    for m in meshes:
        combined_mesh += m

    # Save double-layered sphere
    double_layer_file = os.path.join(output_folder, "double_voronoi_sphere.ply")
    o3d.io.write_triangle_mesh(
        double_layer_file, meshes[0] + meshes[1], write_ascii=True
    )
    print(f"Saved double-layered sphere to {double_layer_file}")

    # Save triple-layered sphere
    triple_layer_file = os.path.join(output_folder, "triple_voronoi_sphere.ply")
    o3d.io.write_triangle_mesh(triple_layer_file, combined_mesh, write_ascii=True)
    print(f"Saved triple-layered sphere to {triple_layer_file}")


if __name__ == "__main__":
    input_file = "data/complex_vsphere/ground_truth/single_voronoi_sphere.ply"
    output_folder = "data/complex_vsphere/ground_truth"

    # Parameters for scaling and rotation
    scales = [0.8, 0.6]  # Scale factors for the inner layers
    rotations = [30, 60]  # Rotation angles for the inner layers (in degrees)

    create_layered_spheres(input_file, output_folder, scales, rotations)
