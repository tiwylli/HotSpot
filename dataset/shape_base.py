import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import scipy.stats
from abc import ABC, abstractmethod
import scipy
from tqdm import tqdm


class ShapeBase(data.Dataset):
    def __init__(
        self,
        n_points,
        n_samples=128,
        grid_res=128,
        grid_range=1.2,
        sample_type="grid",
        sampling_std=0.005,
        n_random_samples=1024,
        resample=True,
        dim=3,
    ):
        # print("n_points:", n_points)
        # print("n_samples:", n_samples)
        # print("grid_res:", grid_res)
        # print("sample_type:", sample_type)
        # print("sampling_std:", sampling_std)
        # print("n_random_sample:", n_random_samples)
        # print("grid_range:", grid_range)
        # print("resample:", resample)
        # print("dim:", dim)

        self.n_points = n_points
        self.n_samples = n_samples
        self.grid_res = grid_res
        self.grid_range = grid_range
        self.sample_type = sample_type
        self.sampling_std = sampling_std
        self.n_random_samples = n_random_samples
        self.resample = resample
        self.dim = dim

        self._init_mnfld_and_grid_points()
        self._resample()

    @abstractmethod
    def get_mnfld_points(self):
        # Implement a function that returns points on the manifold
        # Should return a numpy array of shape (n_points, dim)
        pass

    @abstractmethod
    def get_mnfld_normals(self):
        # Implement a function that returns normal vectors for points on the manifold
        # Should return a numpy array of shape (n_points, dim)
        pass

    @abstractmethod
    def get_points_distances_and_normals(self, points):
        # Implement a function that computes the distance and normal vectors of nonmanifold points.
        # For input points of shape (n_points, dim), the output should be:
        # distances: (n_points, 1)
        # normals: (n_points, dim)

        # Default implementation finds the nearest neighbor and return its normal and the distance to it, which is a coarse approxiamation

        distances = []
        normals = []
        # compute distance and normal (general case)
        kdtree = spatial.cKDTree(self.mnfld_points)
        distances, nn_idx = kdtree.query(points, k=1)
        signs = np.sign(
            np.einsum("ij,ij->i", points - self.mnfld_points[nn_idx], self.mnfld_normals[nn_idx])
        )
        distances = signs * distances
        distances = distances[..., None]
        normals = self.mnfld_normals[nn_idx]

        return distances, normals

    def _init_mnfld_and_grid_points(self):
        self.mnfld_points = self.get_mnfld_points()
        self.mnfld_normals = self.get_mnfld_normals()

        x, y = np.linspace(-self.grid_range, self.grid_range, self.grid_res), np.linspace(
            -self.grid_range, self.grid_range, self.grid_res
        )
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.ravel(), yy.ravel()
        if self.dim == 3:
            z = np.linspace(-self.grid_range, self.grid_range, self.grid_res)
            zz = np.meshgrid(z)
            zz = zz.ravel()
            self.grid_points = np.stack([xx, yy, zz], axis=1).astype("f")
        elif self.dim == 2:
            self.grid_points = np.stack([xx, yy], axis=1).astype("f")
        else:
            raise ValueError("Invalid dimension of dataset")
        # self.grid_dist, self.grid_n = self.get_points_distances_and_normals(self.grid_points)
        # self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])

    def get_nonmnfld_points_and_pdfs(self, sample_type=None, n_nonmnfld_samples=None):
        # Default pdf is uniform
        nonmnfld_pdfs = 1 / (4 * self.grid_range**2)
        nonmnfld_pdfs = np.array([nonmnfld_pdfs])

        if sample_type is None:
            sample_type = self.sample_type
        if n_nonmnfld_samples is None:
            n_nonmnfld_samples = self.n_random_samples

        if sample_type == "grid":
            nonmnfld_points = self.grid_points
            nonmnfld_pdfs = np.ones(nonmnfld_points.shape[:-1] + (1,)) / (4 * self.grid_range**2)
        elif sample_type == "uniform":
            nonmnfld_points = np.random.uniform(
                -self.grid_range, self.grid_range, size=(self.n_random_samples, self.dim)
            ).astype(np.float32)
            nonmnfld_pdfs = np.ones(nonmnfld_points.shape[:-1] + (1,)) / (4 * self.grid_range**2)
        elif sample_type == "central_gaussian":
            nonmnfld_points = np.random.multivariate_normal(
                np.zeros(self.dim),
                np.eye(self.dim) * self.sampling_std**self.dim,
                size=(self.n_random_samples),
            ).astype(np.float32)
            nonmnfld_pdfs = scipy.stats.multivariate_normal.pdf(
                nonmnfld_points,
                mean=np.zeros(self.dim),
                cov=np.eye(self.dim) * self.sampling_std**self.dim,
            ).reshape([-1, 1])
        elif sample_type == "grid_central_gaussian":
            nonmnfld_points1, nonmnfld_pdfs1 = self.get_nonmnfld_points_and_pdfs(sample_type="grid")
            nonmnfld_points2, nonmnfld_pdfs2 = self.get_nonmnfld_points_and_pdfs(
                sample_type="central_gaussian"
            )
            nonmnfld_points = np.concatenate([nonmnfld_points1, nonmnfld_points2], axis=0)
            nonmnfld_pdfs = self._scale_nonmnfld_pdfs([nonmnfld_pdfs1, nonmnfld_pdfs2])
        elif sample_type == "uniform_central_gaussian":
            nonmnfld_points1, nonmnfld_pdfs1 = self.get_nonmnfld_points_and_pdfs(
                sample_type="uniform"
            )
            nonmnfld_points2, nonmnfld_pdfs2 = self.get_nonmnfld_points_and_pdfs(
                sample_type="central_gaussian"
            )
            nonmnfld_points = np.concatenate([nonmnfld_points1, nonmnfld_points2], axis=0)
            nonmnfld_pdfs = self._scale_nonmnfld_pdfs([nonmnfld_pdfs1, nonmnfld_pdfs2])
        # !!! DO NOT USE THE FOLLOWING SAMPLE TYPES !!!
        elif sample_type == "gaussian":
            nonmnfld_points, nonmnfld_pdfs = self._sample_gaussian_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(
                range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
            )
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
            nonmnfld_pdfs = nonmnfld_pdfs[sample_idx, idx]
        elif sample_type == "combined":
            nonmnfld_points = self.grid_points
            nonmnfld_pdfs = np.ones(nonmnfld_points.shape[:-1] + (1,)) / (4 * self.grid_range**2)
            nonmnfld_points1, nonmnfld_pdfs1 = self._sample_gaussian_noise_around_shape()
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
        elif sample_type == "laplace":
            nonmnfld_points = self.sample_laplace_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(
                range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res
            )
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
            self.nonmnfld_probs = self.nonmnfld_probs[sample_idx, idx]
            nonmnfld_pdfs = 0
        elif sample_type == "central_laplace":
            nonmnfld_points, nonmnfld_pdfs = self._sample_central_laplace(
                n_random_sample=self.n_random_samples
            )
        else:
            raise Warning("Unsupported non manfold sampling type {}".format(self.sample_type))
        return nonmnfld_points, nonmnfld_pdfs

    def _scale_nonmnfld_pdfs(self, nonmnfld_pdfs_list):
        # nonmnfld_pdfs_list: list of shape (n_nonmnfld_points, 1)
        n_mix = len(nonmnfld_pdfs_list)
        n_points_list = list(map(lambda x: x.shape[0], nonmnfld_pdfs_list))
        weights = np.array(n_points_list) / np.sum(n_points_list)
        nonmnfld_pdfs_list = [nonmnfld_pdfs_list[i] * weights[i] for i in range(n_mix)]
        nonmnfld_pdfs = np.concatenate(nonmnfld_pdfs_list, axis=0)

        return nonmnfld_pdfs

    def _sample_central_gaussian(self, n_random_samples=None):
        if n_random_samples is None:
            n_random_samples = self.n_random_samples

        nonmnfld_points = np.random.multivariate_normal(
            np.zeros(self.dim),
            np.eye(self.dim) * self.sampling_std**self.dim,
            size=(n_random_samples),
        ).astype(np.float32)
        nonmnfld_pdfs = scipy.stats.multivariate_normal.pdf(
            nonmnfld_points,
            mean=np.zeros(self.dim),
            cov=np.eye(self.dim) * self.sampling_std**self.dim,
            
        ).reshape([-1, 1])

        return nonmnfld_points, nonmnfld_pdfs

    def _sample_central_laplace(self, n_random_samples=None):
        if n_random_samples is None:
            n_random_samples = self.n_random_samples

        laplace_samples = np.random.laplace(0, self.sampling_std, size=(n_random_samples)).astype(
            np.float32
        )  # shape: (n_random_samples,)
        laplace_pdfs = scipy.stats.laplace.pdf(
            laplace_samples, mean=0, cov=self.sampling_std
        )  # shape: (n_random_samples,)

        angle_samples = np.random.uniform(0, 2 * np.pi, size=(n_random_samples)).astype(
            np.float32
        )  # shape: (n_random_samples,)
        angle_pdfs = np.ones(n_random_samples) / (2 * np.pi)  # shape: (n_random_sample,)

        nonmnfld_points = np.concatenate(
            [laplace_samples * np.cos(angle_samples), laplace_samples * np.sin(angle_samples)],
            axis=-1,
        )  # shape: (n_random_sample, 2)
        nonmnfld_pdfs = laplace_pdfs * angle_pdfs  # shape: (n_random_sample,)
        nonmnfld_pdfs = np.expand_dims(nonmnfld_pdfs, axis=-1)  # shape: (n_random_sample, 1)

        return nonmnfld_points, nonmnfld_pdfs

    def _sample_gaussian_noise_around_shape(self):
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

    def _resample(self):
        self.nonmnfld_points, self.nonmnfld_pdfs = self.get_nonmnfld_points_and_pdfs()

        self.nonmnfld_dist_gt, self.nonmnfld_normals_gt = self.get_points_distances_and_normals(
            self.nonmnfld_points
        )

        # self.dist_img = np.reshape(self.grid_dist, [self.grid_res, self.grid_res])

    def __getitem__(self, index):
        if self.resample:
            self._resample()

        mnfld_idx = np.random.permutation(range(self.mnfld_points.shape[0]))
        nonmnfld_idx = np.random.permutation(range(self.nonmnfld_points.shape[0]))

        if self.nonmnfld_pdfs.shape[0] > 1:
            nonmnfld_pdfs = self.nonmnfld_pdfs[nonmnfld_idx]
        else:
            nonmnfld_pdfs = self.nonmnfld_pdfs

        ret_dist = {
            "mnfld_points": self.mnfld_points[mnfld_idx],  # (n_points, dim)
            "mnfld_normals_gt": self.mnfld_normals[mnfld_idx],  # (n_points, dim)
            "nonmnfld_points": self.nonmnfld_points[nonmnfld_idx],  # (n_nonmnfld_samples, dim)
            "nonmnfld_pdfs": nonmnfld_pdfs,  # (n_nonmnfld_samples, 1)
        }

        if self.nonmnfld_dist_gt is not None:
            ret_dist["nonmnfld_dists_gt"] = self.nonmnfld_dist_gt[nonmnfld_idx]
        if self.nonmnfld_normals_gt is not None:
            ret_dist["nonmnfld_normals_gt"] = self.nonmnfld_normals_gt[nonmnfld_idx]

        return ret_dist

    def __len__(self):
        return self.n_samples
