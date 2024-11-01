# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import os.path
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from abc import ABC, abstractmethod
import torch
import open3d as o3d
import scipy
from dataset.shape_base import ShapeBase


class ReconDataset(ShapeBase):
    def __init__(
        self,
        file_path,
        n_points,
        n_samples=128,
        grid_res=128,
        sample_type="grid",
        sampling_std=0.005,
        n_random_samples=4096,
        resample=True,
        requires_dist=False,
        requires_curvatures=False,
        grid_range=1.1,
        compute_sal_dist_gt=False,
        scale_method="default",
    ):
        self.file_path = file_path
        self.requires_dist = requires_dist
        self.requires_curvatures = requires_curvatures
        self.scale_method = scale_method
        # assumes a subdirectory names "estimated props" in dataset path
        self.nonmnfld_dist, self.nonmnfld_n, self.mnfld_curvs = None, None, None
        # Load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)

        super().__init__(
            n_points=n_points,
            n_samples=n_samples,
            grid_res=grid_res,
            grid_range=grid_range,
            sample_type=sample_type,
            sampling_std=sampling_std,
            n_random_samples=n_random_samples,
            resample=resample,
            dim=3,
            compute_sal_dist_gt=compute_sal_dist_gt,
        )

        # extract center and scale points and normals
        self.bbox = np.array([np.min(self.mnfld_points, axis=0), np.max(self.mnfld_points, axis=0)]).transpose()

    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        # center and scale point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        # self.scale = np.linalg.norm(points, axis=-1).max(-1)
        if self.scale_method == "default":
            self.scale = np.abs(points).max()
        elif self.scale_method == "mean":
            # scale such that 80% of the points are within the sphere with radius 0.3
            self.scale = np.percentile(np.linalg.norm(points, axis=-1), 70) / 0.5
            self.scale = max(self.scale, np.abs(points).max())
        points = points / self.scale

        return points

    def get_cp_and_scale(self, scale_method):
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        cp = points.mean(axis=0)
        points = points - cp[None, :]
        if scale_method == "default":
            scale = np.abs(points).max()
        elif scale_method == "mean":
            scale = np.percentile(np.linalg.norm(points, axis=-1), 70) / 0.5
            scale = max(scale, np.abs(points).max())
        return cp, scale

    def get_mnfld_normals(self):
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)

        return normals

    def get_points_distances_and_normals(self, points):
        return None, None

    def _init_mnfld_and_grid_points(self):
        self.mnfld_points = self.get_mnfld_points()
        self.mnfld_normals = self.get_mnfld_normals()

        x, y, z = (
            np.linspace(-self.grid_range, self.grid_range, self.grid_res),
            np.linspace(-self.grid_range, self.grid_range, self.grid_res),
            np.linspace(-self.grid_range, self.grid_range, self.grid_res),
        )
        xx, yy, zz = np.meshgrid(x, y, z)
        xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
        self.grid_points = np.stack([xx, yy, zz], axis=1).astype("f")
        # self.grid_dist, self.grid_n = self.get_points_distances_and_normals(self.grid_points)
        # self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])
