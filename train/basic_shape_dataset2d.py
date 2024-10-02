# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import utils.visualizations as vis
from abc import ABC, abstractmethod
from matplotlib.path import Path
import torch
import scipy


class BasicShape2D(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled on Jets + samples no a grid with their distance to the surface
    def __init__(
        self,
        n_points,
        n_samples=128,
        grid_res=128,
        sample_type="grid",
        sampling_std=0.005,
        n_random_samples=1024,
        grid_range=1.2,
        batch_size=32,
    ):

        print("n_points:", n_points)
        print("n_samples:", n_samples)
        print("grid_res:", grid_res)
        print("sample_type:", sample_type)
        print("sampling_std:", sampling_std)
        print("n_random_sample:", n_random_samples)
        print("batch_size:", batch_size)
        print("grid_range:", grid_range)
        
        self.grid_range = grid_range
        self.n_points = n_points
        self.n_samples = n_samples
        self.grid_res = grid_res
        self.sample_type = sample_type  # grid | gaussian | combined
        self.sampling_std = sampling_std
        self.n_random_samples = n_random_samples
        self.batch_size = batch_size
        # Generate shape

        n_samples = self.n_points
        if "grid" in self.sample_type:
            n_samples += self.grid_res**2
        if self.sample_type != "grid":
            n_samples += self.n_random_samples

        self.n_batchs = int(np.ceil(n_samples / self.batch_size))

        self.points = self.get_mnfld_points()

        # generate grid points and find distance to closest point on the line
        x, y = np.linspace(-grid_range, grid_range, self.grid_res), np.linspace(
            -grid_range, grid_range, self.grid_res
        )
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.ravel(), yy.ravel()
        self.grid_points = np.stack([xx, yy], axis=1).astype("f")
        self.nonmnfld_points, self.nonmnfld_pdfs = self.get_nonmnfld_points_and_pdfs()

        # Compute gt mnfld normals
        self.mnfld_n = self.get_mnfld_n()

        self.grid_dist, self.grid_n = self.get_points_distances_and_normals(self.grid_points)
        # self.nonmnfld_dist, self.nonmnfld_n = self.get_points_distances_and_normals(
        #     self.nonmnfld_points
        # ) # For grid sample, shape: (grid_res ** 2,), (grid_res ** 2, 2)
        self.nonmnfld_dist, self.nonmnfld_n = self.get_points_distances_and_normals(
            self.nonmnfld_points.reshape(-1, self.nonmnfld_points.shape[-1])
        )
        self.nonmnfld_dist = np.reshape(self.nonmnfld_dist, self.nonmnfld_points.shape[:-1])
        self.nonmnfld_n = np.reshape(self.nonmnfld_n, self.nonmnfld_points.shape)

        self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])

        self.point_idxs = np.arange(self.points.shape[1])
        self.grid_points_idxs = np.arange(self.grid_points.shape[0])
        self.nonmnfld_points_idxs = np.arange(self.nonmnfld_points.shape[-2])
        self.sample_probs = np.ones_like(self.grid_points_idxs) / self.grid_points.shape[0]

        self.generate_batch_indices()

    @abstractmethod
    def get_mnfld_points(self):
        # implement a function that returns points on the manifold
        pass

    @abstractmethod
    def get_mnfld_n(self):
        # implement a function that returns normal vectors for points on the manifold
        pass

    @abstractmethod
    def get_points_distances_and_normals(self, points):
        # implement a function that computes the distance and normal vectors of nonmanifold points.
        # default implementation finds the nearest neighbor and return its normal and the distance to it.
        # which is a coarse approxiamation

        distances = []
        normals = []
        # compute distance and normal (general case)
        for i, point_cloud in enumerate(self.points):
            kdtree = spatial.cKDTree(point_cloud)
            distances, nn_idx = kdtree.query(points, k=1)
            signs = np.sign(
                np.einsum("ij,ij->i", points - point_cloud[nn_idx], self.mnfld_n[i, nn_idx])
            )
            normals.append(self.mnfld_n[i, nn_idx])
            distances.append(signs * distances)

        distances = np.stack(distances).astype("f")
        normals = np.stack(normals).astype("f")
        return distances, normals

    def get_grid_divergence(self):
        # 2D implementation
        n_img = np.reshape(self.grid_n, [self.grid_res, self.grid_res, -1])
        frac_45 = 1.0 / np.sqrt(2)
        filter = np.array(
            [
                [[frac_45, -frac_45], [1.0, 0.0], [frac_45, frac_45]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                [[-frac_45, -frac_45], [-1.0, 0.0], [-frac_45, frac_45]],
            ]
        )  # [y, x]
        padding = self.get_padding(n_img, filter, strides=[1, 1])
        n_img = torch.nn.functional.pad(torch.tensor(n_img, dtype=torch.float32), padding)
        div_img = (
            torch.nn.functional.conv2d(
                n_img.permute([2, 0, 1]).unsqueeze(0),
                torch.tensor(filter, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0),
            )
            .squeeze()
            .numpy()
        )

        return div_img.flatten()

    def get_offgrid_divergnce(self, off_grid_points, method="nn"):
        # TODO implement interpulation method?
        if method == "nn":
            # find the nearest grid point and return its divergence
            kdtree = spatial.cKDTree(self.grid_points)
            _, nn_idx = kdtree.query(off_grid_points, k=1)
        else:
            raise Warning("unsupported offgrid div computeation method")
        return self.grid_div[nn_idx]

    def get_padding(self, img, filter, strides=[1, 1]):
        # from https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/3
        in_height, in_width, _ = img.shape
        filter_height, filter_width, _ = filter.shape
        # The total padding applied along the height and width is computed as:
        if in_height % strides[0] == 0:
            pad_along_height = max(filter_height - strides[0], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[0]), 0)
        if in_width % strides[1] == 0:
            pad_along_width = max(filter_width - strides[1], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[1]), 0)

        # Finally, the padding on the top, bottom, left and right are:
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return (0, 0, pad_left, pad_right, pad_top, pad_bottom)

    def get_nonmnfld_points_and_pdfs(self):
        nonmnfld_pdfs = 1 / (4 * self.grid_range**2)
        nonmnfld_pdfs = np.array([nonmnfld_pdfs])
        if self.sample_type == "grid":
            nonmnfld_points = self.grid_points
        elif self.sample_type == "uniform":
            nonmnfld_points = np.random.uniform(
                -self.grid_range, self.grid_range, size=(self.n_random_samples, 2)
            ).astype(np.float32)
        elif self.sample_type == "gaussian":
            nonmnfld_points, nonmnfld_pdfs = self.sample_gaussian_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(
                range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
            )
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
            nonmnfld_pdfs = nonmnfld_pdfs[sample_idx, idx]
        elif self.sample_type == "combined":
            nonmnfld_points = self.grid_points
            nonmnfld_pdfs = np.ones(nonmnfld_points.shape[:-1] + (1,)) / (4 * self.grid_range**2)
            nonmnfld_points1, nonmnfld_pdfs1 = self.sample_gaussian_noise_around_shape()
            n_points1 = nonmnfld_points1.shape[1]
            nonmnfld_points2 = self.grid_points
            nonmnfld_pdfs2 = np.ones(nonmnfld_points2.shape[:-1] + (1,)) / (4 * self.grid_range**2)
            n_points2 = nonmnfld_points2.shape[0]
            # print(n_points1, n_points2)
            nonmnfld_pdfs1 *= n_points1 / (
                n_points1 + n_points2
            )  # shape: (n_samples, n_points1 * n_noisy_points, 1)
            nonmnfld_pdfs2 *= n_points2 / (
                n_points1 + n_points2
            )  # shape: (n_samples, n_points2, 1)
            idx1 = np.random.choice(
                range(nonmnfld_points1.shape[1]), int(np.ceil(self.grid_res * self.grid_res / 2))
            )
            idx2 = np.random.choice(
                range(nonmnfld_points2.shape[0]), int(np.floor(self.grid_res * self.grid_res / 2))
            )
            sample_idx = np.random.choice(
                range(nonmnfld_points1.shape[0]), int(np.ceil(self.grid_res * self.grid_res / 2))
            )

            nonmnfld_points = np.concatenate(
                [nonmnfld_points1[sample_idx, idx1], nonmnfld_points2[idx2]], axis=0
            )
            nonmnfld_pdfs = np.concatenate(
                [nonmnfld_pdfs1[sample_idx, idx1], nonmnfld_pdfs2[idx2]], axis=0
            )
        elif self.sample_type == "laplace":
            nonmnfld_points = self.sample_laplace_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(
                range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
            )
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
            self.nonmnfld_probs = self.nonmnfld_probs[sample_idx, idx]
            nonmnfld_pdfs = 0
        elif self.sample_type == "central_gaussian":
            n_samples = self.grid_res * self.grid_res
            nonmnfld_points = np.random.multivariate_normal(
                [0, 0],
                [[self.sampling_std, 0], [0, self.sampling_std]],
                size=(self.n_samples, n_samples),
            ).astype(np.float32)
            nonmnfld_pdfs = scipy.stats.multivariate_normal.pdf(
                nonmnfld_points, mean=[0, 0], cov=[[self.sampling_std, 0], [0, self.sampling_std]]
            ).reshape([self.n_samples, -1, 1])
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(
                range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
            )
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
            nonmnfld_pdfs = nonmnfld_pdfs[sample_idx, idx]
        elif self.sample_type == "grid_central_gaussian":
            nonmnfld_points1, nonmnfld_pdfs1 = self.sample_central_gaussian(
                n_random_sample=self.n_random_samples
            )  # shape: (n_samples, n_random_samples, 2), (n_samples, n_random_samples, 1)
            n_points1 = nonmnfld_points1.shape[1]
            # print(nonmnfld_points1.shape)
            # print(nonmnfld_pdfs1.shape)

            nonmnfld_points2 = self.grid_points[None, ...].repeat(
                self.n_samples, axis=0
            )  # shape: (n_samples, grid_res ** 2, 2)
            nonmnfld_pdfs2 = np.ones(nonmnfld_points2.shape[:-1] + (1,)) / (
                4 * self.grid_range**2
            )  # shape: (n_samples, grid_res ** 2, 1)
            n_points2 = nonmnfld_points2.shape[1]
            # print(nonmnfld_points2.shape)
            # print(nonmnfld_pdfs2.shape)

            nonmnfld_pdfs1 *= n_points1 / (
                n_points1 + n_points2
            )  # shape: (n_samples, n_points1 * n_noisy_points, 1)
            nonmnfld_pdfs2 *= n_points2 / (
                n_points1 + n_points2
            )  # shape: (n_samples, n_points2, 1)

            nonmnfld_points = np.concatenate(
                [nonmnfld_points1, nonmnfld_points2], axis=1
            )  # shape: (n_samples, n_random_samples + grid_res ** 2, 2)
            nonmnfld_pdfs = np.concatenate(
                [nonmnfld_pdfs1, nonmnfld_pdfs2], axis=1
            )  # shape: (n_samples, n_random_samples + grid_res ** 2, 1)
            # n_points = nonmnfld_points.shape[1]

            # # randomly sample points from the combined set for every sample
            # idx = np.random.choice(range(n_points), int(n_points))

            # nonmnfld_points = np.concatenate(
            #     [nonmnfld_points1[sample_idx, idx1], nonmnfld_points2[idx2]], axis=0
            # )
            # nonmnfld_pdfs = np.concatenate(
            #     [nonmnfld_pdfs1[sample_idx, idx1], nonmnfld_pdfs2[idx2]], axis=0
            # )
        elif self.sample_type == "central_laplace":
            nonmnfld_points, nonmnfld_pdfs = self.sample_central_laplace(
                n_random_sample=self.n_random_samples
            )
        else:
            raise Warning("Unsupported non manfold sampling type {}".format(self.sample_type))
        return nonmnfld_points, nonmnfld_pdfs

    def sample_gaussian_noise_around_shape(self):
        n_noisy_points = int(np.round(self.grid_res * self.grid_res / self.n_points))
        noise = np.random.multivariate_normal(
            [0, 0],
            [[self.sampling_std, 0], [0, self.sampling_std]],
            size=(self.n_samples, self.n_points, n_noisy_points),
        ).astype(np.float32)
        nonmnfld_pdfs = scipy.stats.multivariate_normal.pdf(
            noise, mean=[0, 0], cov=[[self.sampling_std, 0], [0, self.sampling_std]]
        )
        nonmnfld_pdfs = nonmnfld_pdfs.reshape(
            [self.n_samples, -1, 1]
        )  # shape: (n_samples, n_points * n_noisy_points, 1)
        nonmnfld_points = np.tile(self.points[:, :, None, :], [1, 1, n_noisy_points, 1]) + noise
        nonmnfld_points = nonmnfld_points.reshape(
            [nonmnfld_points.shape[0], -1, nonmnfld_points.shape[-1]]
        )  # shape: (n_samples, n_points * n_noisy_points, 2)
        return nonmnfld_points, nonmnfld_pdfs

    def sample_central_gaussian(self, n_random_sample):
        nonmnfld_points = np.random.multivariate_normal(
            [0, 0],
            [[self.sampling_std, 0], [0, self.sampling_std]],
            size=(self.n_samples, n_random_sample),
        ).astype(
            np.float32
        )  # shape: (n_samples, n_random_sample, 2)
        nonmnfld_pdfs = scipy.stats.multivariate_normal.pdf(
            nonmnfld_points, mean=[0, 0], cov=[[self.sampling_std, 0], [0, self.sampling_std]]
        )  # shape: (n_samples, n_random_sample)
        nonmnfld_pdfs = np.expand_dims(
            nonmnfld_pdfs, axis=-1
        )  # shape: (n_samples, n_random_sample, 1)
        # idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
        # sample_idx = np.random.choice(
        #     range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
        # )
        # nonmnfld_points = nonmnfld_points[sample_idx, idx]
        # nonmnfld_pdfs = nonmnfld_pdfs[sample_idx, idx]

        return nonmnfld_points, nonmnfld_pdfs

    def sample_central_laplace(self, n_random_sample):
        laplace_samples = np.random.laplace(0, self.sampling_std, size=(self.n_samples, n_random_sample)).astype(np.float32)  # shape: (n_samples, n_random_sample)
        laplace_pdfs = scipy.stats.laplace.pdf(
            laplace_samples, mean=0, cov=self.sampling_std
        )  # shape: (n_samples, n_random_sample)

        angle_samples = np.random.uniform(0, 2 * np.pi, size=(self.n_samples, n_random_sample)).astype(np.float32)  # shape: (n_samples, n_random_sample)
        angle_pdfs = np.ones((self.n_samples, n_random_sample)) / (2 * np.pi)  # shape: (n_samples, n_random_sample)        

        nonmnfld_points = np.concatenate(
            [laplace_samples * np.cos(angle_samples), laplace_samples * np.sin(angle_samples)], axis=-1
        )  # shape: (n_samples, n_random_sample, 2)
        nonmnfld_pdfs = laplace_pdfs * angle_pdfs  # shape: (n_samples, n_random_sample)

        nonmnfld_pdfs = np.expand_dims(
            nonmnfld_pdfs, axis=-1
        )  # shape: (n_samples, n_random_sample, 1)
        # idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
        # sample_idx = np.random.choice(
        #     range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
        # )
        # nonmnfld_points = nonmnfld_points[sample_idx, idx]
        # nonmnfld_pdfs = nonmnfld_pdfs[sample_idx, idx]

        return nonmnfld_points, nonmnfld_pdfs

    def sample_laplace_noise_around_shape(self, heat_lambda=100):
        n_noisy_points = int(np.round(self.grid_res * self.grid_res / self.n_points))
        laplace_noise = np.random.laplace(
            0, 1 / heat_lambda, size=(self.n_samples, self.n_points, n_noisy_points, 1)
        ).astype(np.float32)
        angle = np.random.uniform(
            0, 2 * np.pi, size=(self.n_samples, self.n_points, n_noisy_points, 1)
        )
        noise = np.concatenate(
            [laplace_noise * np.cos(angle), laplace_noise * np.sin(angle)], axis=-1
        )  # shape: (n_samples, n_points, n_noisy_points, 2)
        nonmnfld_points = np.tile(self.points[:, :, None, :], [1, 1, n_noisy_points, 1]) + noise
        nonmnfld_points = nonmnfld_points.reshape(
            [nonmnfld_points.shape[0], -1, nonmnfld_points.shape[-1]]
        )

        self.nonmnfld_probs = (heat_lambda / 2) * np.exp(
            -np.abs(laplace_noise) * heat_lambda
        )  # heat_lambda = 1 / laplace_lambda
        self.nonmnfld_probs *= 1 / (2 * np.pi)
        self.nonmnfld_probs = self.nonmnfld_probs.reshape([self.n_samples, -1, 1])

        return nonmnfld_points

    def resample(self):
        self.nonmnfld_points, self.nonmnfld_pdfs = self.get_nonmnfld_points_and_pdfs()

        # Compute gt mnfld normals
        self.mnfld_n = self.get_mnfld_n()

        self.grid_dist, self.grid_n = self.get_points_distances_and_normals(self.grid_points)
        # self.nonmnfld_dist, self.nonmnfld_n = self.get_points_distances_and_normals(
        #     self.nonmnfld_points
        # ) # For grid sample, shape: (grid_res ** 2,), (grid_res ** 2, 2)
        self.nonmnfld_dist, self.nonmnfld_n = self.get_points_distances_and_normals(
            self.nonmnfld_points.reshape(-1, self.nonmnfld_points.shape[-1])
        )
        self.nonmnfld_dist = np.reshape(self.nonmnfld_dist, self.nonmnfld_points.shape[:-1])
        self.nonmnfld_n = np.reshape(self.nonmnfld_n, self.nonmnfld_points.shape)

        self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])

        self.point_idxs = np.arange(self.points.shape[1])
        self.grid_points_idxs = np.arange(self.grid_points.shape[0])
        self.nonmnfld_points_idxs = np.arange(self.nonmnfld_points.shape[-2])
        self.sample_probs = np.ones_like(self.grid_points_idxs) / self.grid_points.shape[0]

        self.generate_batch_indices()

    def generate_batch_indices(self):
        mnfld_idx = []
        nonmnfld_idx = []
        for i in range(self.n_samples):
            mnfld_idx.append(np.random.choice(self.point_idxs, self.n_points))
            nonmnfld_idx.append(np.random.choice(self.nonmnfld_points_idxs, min(self.n_points, len(self.nonmnfld_points_idxs))))
        self.mnfld_idx = np.array(mnfld_idx)
        self.nonmnfld_idx = np.array(nonmnfld_idx)

    def __getitem__(self, index):
        # nonmnfld_idx = self.nonmnfld_idx[index]
        # mnfld_idx = self.mnfld_idx[index]

        # if self.nonmnfld_dist is not None:
        #     nonmnfld_dist = self.nonmnfld_dist[index, nonmnfld_idx]
        # else:
        #     nonmnfld_dist = torch.tensor(0)

        # if self.nonmnfld_n is not None:
        #     nonmnfld_n = self.nonmnfld_n[index, nonmnfld_idx]
        # else:
        #     nonmnfld_n = torch.tensor(0)
        mnfld_points = self.get_mnfld_points()
        mnfld_normals = self.get_mnfld_n()

        nonmnfld_points, nonmnfld_pdfs = self.get_nonmnfld_points_and_pdfs()
        nonmnfld_dist, nonmnfld_n = self.get_points_distances_and_normals(
            nonmnfld_points.reshape(-1, nonmnfld_points.shape[-1])
        )
        nonmnfld_dist = np.reshape(nonmnfld_dist, nonmnfld_points.shape[:-1])
        nonmnfld_n = np.reshape(nonmnfld_n, nonmnfld_points.shape)
        
        # if len(self.nonmnfld_points.shape) == 2:
        #     nonmnfld_points = self.nonmnfld_points[nonmnfld_idx]
        #     nonmnfld_pdfs = self.nonmnfld_pdfs[nonmnfld_idx]
        #     nonmnfld_dist = self.nonmnfld_dist[nonmnfld_idx]
        #     nonmnfld_n = self.nonmnfld_n[nonmnfld_idx]
        # else:
        #     nonmnfld_points = self.nonmnfld_points[index, nonmnfld_idx]
        #     nonmnfld_pdfs = self.nonmnfld_pdfs[index, nonmnfld_idx]
        #     nonmnfld_dist = self.nonmnfld_dist[index, nonmnfld_idx]
        #     nonmnfld_n = self.nonmnfld_n[index, nonmnfld_idx]

        # print(f"self.points.shape: {self.points.shape}")
        # print(f"self.mnfld_n.shape: {self.mnfld_n.shape}")
        # print(f"self.nonmnfld_dist.shape: {self.nonmnfld_dist.shape}")
        # print(f"self.nonmnfld_n.shape: {self.nonmnfld_n.shape}")
        # print(f"self.nonmnfld_points.shape: {self.nonmnfld_points.shape}")
        # print(f"self.nonmnfld_pdfs.shape: {self.nonmnfld_pdfs.shape}")

        return {
            "mnfld_points": mnfld_points,
            "mnfld_normals": mnfld_normals,
            "nonmnfld_dist": nonmnfld_dist,
            "nonmnfld_normals": nonmnfld_n,
            "nonmnfld_points": nonmnfld_points,
            "nonmnfld_pdfs": nonmnfld_pdfs,
        }

    def __len__(self):
        return self.n_samples


class Circle(BasicShape2D):
    def __init__(self, *args, r=0.5):
        self.r = r
        BasicShape2D.__init__(self, *args)

    def get_mnfld_points(self):
        theta = np.random.uniform(0, 2 * np.pi, size=(self.n_samples, self.n_points)).astype("f")
        x = self.r * np.sin(theta)
        y = self.r * np.cos(theta)

        points = np.stack([x, y], axis=2)
        return points

    def get_mnfld_n(self):
        return self.points / np.linalg.norm(self.points, axis=2, keepdims=True)

    def get_points_distances_and_normals(self, points):
        point_dist = np.linalg.norm(points, axis=1, keepdims=True)
        distances = point_dist - self.r
        normals = points / point_dist
        return distances, normals


class Polygon(BasicShape2D):
    def __init__(self, *args, vertices=[], line_sample_type="uniform"):

        # vertices: x,y points specifying the polygon
        self.vertices = np.array(vertices)
        self.lines = self.get_line_props()
        self.line_sample_type = line_sample_type
        BasicShape2D.__init__(self, *args)

    def get_mnfld_points(self):
        # sample points on the lines
        n_points_to_sample = self.n_points - len(self.vertices)
        if n_points_to_sample < 0:
            raise Warning(
                "Fewer points to sample than polygon vertices. Please change the number of points"
            )
        sample_prob = self.lines["line_length"] / np.sum(self.lines["line_length"])
        points_per_segment = np.floor(n_points_to_sample * sample_prob).astype(np.int32)
        points_leftover = int(n_points_to_sample - points_per_segment.sum())
        if not points_leftover == 0:
            for j in range(points_leftover):
                actual_prob = points_per_segment / points_per_segment.sum()
                prob_diff = sample_prob - actual_prob
                add_idx = np.argmax(prob_diff)
                points_per_segment[add_idx] = points_per_segment[add_idx] + 1

        points = []
        self.point_normal = []
        for point_idx, point in enumerate(self.vertices):
            l1_idx = len(self.vertices) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines["nl"][l1_idx] + self.lines["nl"][l2_idx]
            self.point_normal.append(n / np.linalg.norm(n))
            points.append(point)
        # points = np.repeat(np.array(points)[None, :], self.n_samples, axis=0)
        # self.point_normal = np.repeat(np.array(self.point_normal)[None, :], self.n_samples, axis=0)

        for line_idx in range(len(self.lines["A"])):
            if self.line_sample_type == "uniform":
                t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
                # t = np.repeat(t[None, :], self.n_samples, axis=0)
            else:
                # t = np.random.uniform(0, 1, [self.n_samples, points_per_segment[line_idx]])
                t = np.random.uniform(0, 1, [points_per_segment[line_idx]])
            p1 = np.array(self.vertices[self.lines["start_idx"][line_idx]])
            p2 = np.array(self.vertices[self.lines["end_idx"][line_idx]])
            # points = np.concatenate([points, p1 + t[:, :, None] * (p2 - p1)], axis=1)
            points = np.concatenate([points, p1 + t[:, None] * (p2 - p1)], axis=0)
            self.point_normal = np.concatenate(
                [
                    self.point_normal,
                    np.tile(
                        # self.lines["nl"][line_idx][None, None, :],
                        # [self.n_samples, points_per_segment[line_idx], 1],
                        self.lines["nl"][line_idx][None, :],
                        [points_per_segment[line_idx], 1],
                    ),
                ],
                # axis=1,
                axis=0,
            )
        return points.astype("f")

    def get_mnfld_n(self):
        return self.point_normal

    def get_points_distances_and_normals(self, points):
        # iterate over all the lines and  finds the minimum distance between all points and line segments
        # good explenation ref : https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
        n_grid_points = len(points)
        p1x = np.vstack(self.vertices[self.lines["start_idx"]][:, 0])
        p1y = np.vstack(self.vertices[self.lines["start_idx"]][:, 1])
        p2x = np.vstack(self.vertices[self.lines["end_idx"]][:, 0])
        p2y = np.vstack(self.vertices[self.lines["end_idx"]][:, 1])
        p1p2 = np.array(self.lines["direction"])
        px = points[:, 0]
        py = points[:, 1]
        pp1 = np.vstack(
            [px - np.tile(p1x, [1, 1, n_grid_points]), py - np.tile(p1y, [1, 1, n_grid_points])]
        )
        pp2 = np.vstack(
            [px - np.tile(p2x, [1, 1, n_grid_points]), py - np.tile(p2y, [1, 1, n_grid_points])]
        )

        r = (p1p2[:, 0, None] * pp1[0, :, :] + p1p2[:, 1, None] * pp1[1, :, :]) / np.array(
            self.lines["line_length"]
        )[:, None]

        d1 = np.linalg.norm(pp1, axis=0)
        d2 = np.linalg.norm(pp2, axis=0)
        dp = np.sqrt(np.square(d1) - np.square(r * np.array(self.lines["line_length"])[:, None]))
        d = np.where(r < 0, d1, np.where(r > 1, d2, dp))
        distances = np.min(d, axis=0)
        idx = np.argmin(d, axis=0)
        # compute normal vector
        polygon_path = Path(self.vertices)
        point_in_polygon = polygon_path.contains_points(points)
        point_sign = np.where(point_in_polygon, -1, 1)

        n = np.where(
            r < 0,
            pp1,
            np.where(
                r > 1,
                pp2,
                point_sign
                * np.tile(
                    np.array(self.lines["nl"]).transpose()[:, :, None], [1, 1, n_grid_points]
                ),
            ),
        )
        normals = np.take_along_axis(n, idx[None, None, :], axis=1).squeeze().transpose()
        normals = point_sign[:, None] * normals / np.linalg.norm(normals, axis=1, keepdims=True)
        distances = point_sign * distances

        return distances, normals

    def get_line_props(self):
        lines = {
            "A": [],
            "B": [],
            "C": [],
            "nl": [],
            "line_length": [],
            "start_idx": [],
            "end_idx": [],
            "direction": [],
        }
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices) - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = -(A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [end_point[0] - start_point[0], end_point[1] - start_point[1]] / line_length
            nl = [A, B]
            nl = nl / np.linalg.norm(nl)
            line_props = {
                "A": A,
                "B": B,
                "C": C,
                "nl": nl,
                "line_length": line_length,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "direction": direction,
            }
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines


class Union(BasicShape2D):
    def __init__(self, shapes=[]):
        self.shapes = shapes

        self.n_points = sum([shape.n_points for shape in shapes])
        self.n_samples = shapes[0].n_samples
        self.grid_res = shapes[0].grid_res
        self.sample_type = shapes[0].sample_type
        self.sampling_std = shapes[0].sampling_std
        self.n_random_samples = shapes[0].n_random_samples
        self.grid_range = shapes[0].grid_range
        self.batch_size = shapes[0].batch_size
        
        # BasicShape2D.__init__(self, n_points, n_samples, grid_res, sample_type, sampling_std, n_random_samples, grid_range, batch_size)

    def get_mnfld_points(self):
        points = []
        for shape in self.shapes:
            points.append(shape.get_mnfld_points())
        return np.concatenate(points, axis=-2)
    
    def get_mnfld_n(self):
        normals = []
        for shape in self.shapes:
            normals.append(shape.get_mnfld_n())
        return np.concatenate(normals, axis=-2)

    def get_points_distances_and_normals(self, points):
        distances = []
        normals = []
        for shape in self.shapes:
            d, n = shape.get_points_distances_and_normals(points)
            distances.append(d)
            normals.append(n)
        return np.stack(distances).min(axis=0), np.stack(normals)[np.argmin(np.stack(distances), axis=0), range(points.shape[0])]


def koch_line(start, end, factor):
    """
    Segments a line to Koch line, creating fractals.


    :param tuple start:  (x, y) coordinates of the starting point
    :param tuple end: (x, y) coordinates of the end point
    :param float factor: the multiple of sixty degrees to rotate
    :returns tuple: tuple of all points of segmentation
    """

    # coordinates of the start
    x1, y1 = start[0], start[1]

    # coordinates of the end
    x2, y2 = end[0], end[1]

    # the length of the line
    l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # first point: same as the start
    a = (x1, y1)

    # second point: one third in each direction from the first point
    b = (x1 + (x2 - x1) / 3.0, y1 + (y2 - y1) / 3.0)

    # third point: rotation for multiple of 60 degrees
    c = (
        b[0] + l / 3.0 * np.cos(factor * np.pi / 3.0),
        b[1] + l / 3.0 * np.sin(factor * np.pi / 3.0),
    )

    # fourth point: two thirds in each direction from the first point
    d = (x1 + 2.0 * (x2 - x1) / 3.0, y1 + 2.0 * (y2 - y1) / 3.0)

    # the last point
    e = end

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "factor": factor}


def koch_snowflake(degree, s=1.0):
    """Generates all lines for a Koch Snowflake with a given degree.
    code from: https://github.com/IlievskiV/Amusive-Blogging-N-Coding/blob/master/Visualizations/snowflake.ipynb
    :param int degree: how deep to go in the branching process
    :param float s: the length of the initial equilateral triangle
    :returns list: list of all lines that form the snowflake
    """
    # all lines of the snowflake
    lines = []

    # we rotate in multiples of 60 degrees
    sixty_degrees = np.pi / 3.0

    # vertices of the initial equilateral triangle
    A = (0.0, 0.0)
    B = (s, 0.0)
    C = (s * np.cos(sixty_degrees), s * np.sin(sixty_degrees))

    # set the initial lines
    if degree == 0:
        lines.append(koch_line(A, B, 0))
        lines.append(koch_line(B, C, 2))
        lines.append(koch_line(C, A, 4))
    else:
        lines.append(koch_line(A, B, 5))
        lines.append(koch_line(B, C, 1))
        lines.append(koch_line(C, A, 3))

    for i in range(1, degree):
        # every lines produce 4 more lines
        for _ in range(3 * 4 ** (i - 1)):
            line = lines.pop(0)
            factor = line["factor"]

            lines.append(koch_line(line["a"], line["b"], factor % 6))  # a to b
            lines.append(koch_line(line["b"], line["c"], (factor - 1) % 6))  # b to c
            lines.append(koch_line(line["c"], line["d"], (factor + 1) % 6))  # d to c
            lines.append(koch_line(line["d"], line["e"], factor % 6))  # d to e

    return lines


def get_koch_points(degree, s=1.0):
    lines = koch_snowflake(degree, s=s)
    points = []
    for line in lines:
        for key in line.keys():
            if not key == "factor" and not key == "e":
                points.append(line[key])
    points = np.array(points) - np.array([s / 2, (s / 2) * np.tan(np.pi / 6)])
    points = np.flipud(points)  # reorder the points clockwise
    return points

def get_star_points(transform):
    points = [(1.0, 0.0), (0.4045, 0.2939), (0.3090, 0.9511), (-0.1545, 0.4755), (-0.8090, 0.5878), (-0.5, 0.0), (-0.8090, -0.5878), (-0.1545, -0.4755), (0.3090, -0.9511), (0.4045, -0.2939)]
    # transform points, transform is a 3x3 matrix
    points = np.array(points)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transform, points.T).T
    points = points[:, :2]
    return points

def get_hexagon_points(transform):
    points = [(1.0, 0.0), (0.5000, 0.8660), (-0.5000, 0.8660), (-1.0, 0.0), (-0.5000, -0.8660), (0.5000, -0.8660)]
    points = np.array(points)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transform, points.T).T
    points = points[:, :2]
    return points

def get2D_dataset(*args, shape_type="circle"):

    if shape_type == "circle":
        out_shape = Circle(*args)
    elif shape_type == "L":
        out_shape = Polygon(
            *args,
            vertices=[[0.0, 0.0], [0.5, 0.0], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0, 0.5]]
        )
    elif shape_type == "square":
        out_shape = Polygon(*args, vertices=[[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    elif shape_type == "snowflake":
        vertices = get_koch_points(degree=2, s=1.0)
        out_shape = Polygon(*args, vertices=vertices)
    elif shape_type == "starAndHexagon":
        # new_args = list(args)
        # new_args.n_points //= 2
        transform_star = np.array([[0.5, 0, -0.5], [0, 0.5, -0.5], [0, 0, 1]])        
        transform_hexagon = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]])
        star_points = get_star_points(transform_star)
        hexagon_points = get_hexagon_points(transform_hexagon)
        new_args = list(args)
        new_args[0] = new_args[0] // 2
        out_shape = Union(shapes=[Polygon(*new_args, vertices=star_points), Polygon(*new_args, vertices=hexagon_points)])
    else:
        raise Warning("Unsupportaed shape")

    return out_shape


if __name__ == "__main__":
    np.random.seed(0)
    shape_type = "L"
    res = 128  # has to be even
    example_idx = 0
    sample_type = "grid"
    n_samples = 2
    n_points = 24
    dataset = get2D_dataset(
        n_points, n_samples, res, sample_type, 0.005, shape_type=shape_type
    )  # BasicShape2D(100, 20, res=50)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True
    )
    data = next(iter(dataloader))
    clean_points_gt = data["mnfld_points"][example_idx].detach().cpu().numpy()
    n_gt = data["mnfld_normals"][example_idx].detach().cpu().numpy()
    nonmnfld_points = data["nonmnfld_points"][example_idx].detach().cpu().numpy()
    grid_normals = dataset.grid_n

    vis.plot_sdf_indicator(
        dataset.vertices,
        dataset.grid_points[:, 0],
        dataset.grid_points[:, 1],
        dataset.dist_img.flatten(),
        title_text="",
        show_ax=False,
        output_path="./vis/",
    )  # plot sdf, indicator function, and points

    # vis.plot_shape_data(dataset.grid_points[:, 0], dataset.grid_points[:, 1], dataset.dist_img.flatten(),
    #                     clean_points_gt, n_gt=n_gt, show_ax=True, show_bar=True,
    #                     title_text='', colorscale='Geyser', nonmnfld_points=nonmnfld_points, divergence=None,
    #                     grid_normals=grid_normals) # plot shape, sdf and other data

    # vis.plot_paper_teaser_images(dataset.grid_points[:, 0], dataset.grid_points[:, 1] ,dataset.dist_img.flatten(),
    #                              clean_points_gt, grid_normals) # plot images for DiGS paper teaser image
