import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os
import glob


def fuse_pcd(data_root, scene_title,number_of_points=10**6, ref_title="points.ply"):
    # read reference point cloud
    
    # read mesh
    glb_path = os.path.join(data_root, scene_title)
    mesh = o3d.io.read_triangle_mesh(glb_path, enable_post_processing=True)

    # blender to opencv coordinate system
    blender_to_opencv = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])

    mesh_vertices = np.asarray(mesh.vertices)
    mesh_vertices = mesh_vertices @ blender_to_opencv.T
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    # sample points from mesh
    sampled_pcd = mesh.sample_points_uniformly(number_of_points)

    
    print(np.asarray(sampled_pcd.points).shape)

    # # voxel downsampling
    # down_pcd = sampled_pcd.voxel_down_sample(voxel_size=0.01)
    # down_points = np.asarray(down_pcd.points)
    # print(down_points.shape)

    # # find nearest neighbors for color
    # kdtree = cKDTree(ref_points)
    # dist, idx = kdtree.query(down_points, p=1, workers=-1)
    # down_colors = ref_colors[idx]

    # # update down_pcd colors
    # down_pcd.colors = o3d.utility.Vector3dVector(down_colors)

    # # fuse point clouds
    final_pcd = sampled_pcd
    # final_pcd = final_pcd.voxel_down_sample(voxel_size=0.01)
    # print(final_pcd)

    # # save final point cloud
    o3d.io.write_point_cloud(os.path.join(data_root, ref_title), final_pcd, compressed=True) # save as ply file


if __name__ == "__main__":
    data_root = "./scripts/mesh2pointclouds" # your data root
    scene_title = "xeno_raven.glb" # your mesh file, here I give a example usage
    fuse_pcd(data_root, scene_title)

