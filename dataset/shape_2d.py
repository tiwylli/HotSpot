import numpy as np
from dataset.shape_base import ShapeBase
from matplotlib.path import Path


class Circle(ShapeBase):
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
        outward_normal=True,
        r=0.5,
        center=(0, 0),
    ):
        self.r = r
        self.center = np.array(center, dtype="f")
        self.outward_normal = outward_normal
        ShapeBase.__init__(
            self,
            n_points,
            n_samples,
            grid_res,
            grid_range,
            sample_type,
            sampling_std,
            n_random_samples,
            resample,
            2,
        )

    def get_mnfld_points(self):
        theta = np.random.uniform(0, 2 * np.pi, size=(self.n_points)).astype("f")
        x = self.r * np.sin(theta) + self.center[0]
        y = self.r * np.cos(theta) + self.center[1]

        points = np.stack([x, y], axis=-1)
        return points

    def get_mnfld_normals(self):
        vector = self.mnfld_points - self.center
        vector *= 1 if self.outward_normal else -1
        return vector / np.linalg.norm(vector, axis=-1, keepdims=True)

    def get_points_distances_and_normals(self, points):
        vector = points - self.center
        vector *= 1 if self.outward_normal else -1
        point_dist = np.linalg.norm(vector, axis=-1, keepdims=True)
        distances = point_dist - self.r
        normals = vector / point_dist
        return distances, normals


class Polygon(ShapeBase):
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
        outward_normal=True,  # Clockwise vertices is outward normal by default. This arg will reverse the normal direction
        vertices=[],
        line_sample_type="uniform",
    ):
        # vertices: x,y points specifying the polygon
        self.vertices = np.array(vertices)
        self.lines = self._get_line_props()
        self.line_sample_type = line_sample_type
        self.outward_normal = outward_normal
        ShapeBase.__init__(
            self,
            n_points,
            n_samples,
            grid_res,
            grid_range,
            sample_type,
            sampling_std,
            n_random_samples,
            resample,
            2,
        )

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

    def get_mnfld_normals(self):
        return self.point_normal * (1 if self.outward_normal else -1)

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
        normals *= 1 if self.outward_normal else -1
        distances = point_sign * distances
        distances = distances[:, None]

        return distances, normals

    def _get_line_props(self):
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


class Union(ShapeBase):
    def __init__(self, shapes=[]):
        self.shapes = shapes

        n_points = sum([shape.n_points for shape in shapes])
        n_samples = shapes[0].n_samples
        grid_res = shapes[0].grid_res
        grid_range = shapes[0].grid_range
        sample_type = shapes[0].sample_type
        sampling_std = shapes[0].sampling_std
        n_random_samples = shapes[0].n_random_samples
        resample = shapes[0].resample
        dim = shapes[0].dim

        ShapeBase.__init__(
            self,
            n_points,
            n_samples,
            grid_res,
            grid_range,
            sample_type,
            sampling_std,
            n_random_samples,
            resample,
            dim,
        )

    def get_mnfld_points(self):
        points = []
        for shape in self.shapes:
            points.append(shape.mnfld_points)
        return np.concatenate(points, axis=-2)

    def get_mnfld_normals(self):
        normals = []
        for shape in self.shapes:
            normals.append(shape.mnfld_normals)
        return np.concatenate(normals, axis=-2)

    def get_points_distances_and_normals(self, points):
        distances = []
        normals = []
        for shape in self.shapes:
            d, n = shape.get_points_distances_and_normals(points)
            distances.append(d)
            normals.append(n)
        distances = np.stack(distances)
        normals = np.stack(normals)

        idx = np.argmin(np.abs(distances), axis=0)
        idx = idx.squeeze(-1)
        distances = distances[idx, np.arange(distances.shape[1]), :]
        normals = normals[idx, np.arange(normals.shape[1]), :]

        return (
            distances,
            normals,
        )


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
    points = [
        (0.4045, -0.2939),
        (0.3090, -0.9511),
        (-0.1545, -0.4755),
        (-0.8090, -0.5878),
        (-0.5, 0.0),
        (-0.8090, 0.5878),
        (-0.1545, 0.4755),
        (0.3090, 0.9511),
        (0.4045, 0.2939),
        (1.0, 0.0),
    ]
    # transform points, transform is a 3x3 matrix
    points = np.array(points)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transform, points.T).T
    points = points[:, :2]
    return points

def get_tri_points(center, r):
    points = [
        (1, 0.0),
        (-0.5, -0.866),
        (-0.5, 0.866)
    ]
    points = np.array(points)
    points = points * r + center
    return points

def get_hexagon_points(transform):
    points = [
        (0.5000, -0.8660),
        (-0.5000, -0.8660),
        (-1.0, 0.0),
        (-0.5000, 0.8660),
        (0.5000, 0.8660),
        (1.0, 0.0),
    ]
    points = np.array(points)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transform, points.T).T
    points = points[:, :2]
    return points


def get2D_dataset(
    n_points,
    n_samples=100,
    grid_res=128,
    grid_range=1.2,
    sample_type="grid",
    sampling_std=None,
    n_random_samples=None,
    resample=True,
    shape_type="circle",
):
    args = [
        n_points,
        n_samples,
        grid_res,
        grid_range,
        sample_type,
        sampling_std,
        n_random_samples,
        resample,
    ]

    if shape_type == "circle":
        out_shape = Circle(*args, r=0.5)
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
    elif shape_type == "starhex":
        transform_star = np.array([[0.5, 0, -0.5], [0, 0.5, -0.5], [0, 0, 1]])
        transform_hexagon = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]])
        star_points = get_star_points(transform_star)
        hexagon_points = get_hexagon_points(transform_hexagon)
        args[0] //= 2
        out_shape = Union(
            shapes=[
                Polygon(*args, vertices=star_points),
                Polygon(*args, vertices=hexagon_points, outward_normal=False),
            ]
        )
    elif shape_type == "button":
        args[0] //= 5
        out_shape = Union(
            shapes=[
                Circle(*args, r=1.0, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.2, center=(0.25, 0.25), outward_normal=False),
                Circle(*args, r=0.2, center=(-0.25, 0.25), outward_normal=False),
                Circle(*args, r=0.2, center=(-0.25, -0.25), outward_normal=False),
                Circle(*args, r=0.2, center=(0.25, -0.25), outward_normal=False),
            ]
        )
    elif shape_type == "target":
        args[0] //= 5
        out_shape = Union(
            shapes=[
                Circle(*args, r=0.85, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.75, center=(0, 0), outward_normal=False),
                Circle(*args, r=0.65, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.55, center=(0, 0), outward_normal=False),
                Circle(*args, r=0.45, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.35, center=(0, 0), outward_normal=False),
                Circle(*args, r=0.25, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.15, center=(0, 0), outward_normal=False),
                Circle(*args, r=0.05, center=(0, 0), outward_normal=True)
            ]
        )
    elif shape_type == "bearing":
        args[0] //= 4
        out_shape = Union(
            shapes=[
                Circle(*args, r=1.0, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.75, center=(0, 0), outward_normal=False),
                Circle(*args, r=0.5, center=(0, 0), outward_normal=True),
                Circle(*args, r=0.25, center=(0, 0), outward_normal=False)
            ]
        )
    elif shape_type == "snake":
        points = [
        (0.051245, -0.127382),
        (-0.005640149, -0.24482387),
        (-0.1707916, -0.265009054),
        (-0.35032774, -0.16544293),
        (-0.39891275, 0.03236746),
        (-0.26703915, 0.16771142),
        (-0.083110187, 0.25447037),
        (0.142463076407, 0.2579407),
        (0.3020995400, 0.10871533),
        (0.33680311, -0.109917),
        (0.26045524, -0.3632533),
        (-0.1074026, -0.5124787),
        (-0.444027409, -0.42919),
        (-0.732067, -0.2070872297757),
        (-0.75635, 0.0844228),
        (-0.697363, 0.282233),
        (-0.43708669, 0.587624),
        (0.01821977, 0.767273933),
        (0.545024, 0.6362097),
        (0.850416, 0.1330078),
        (0.926763, -0.29384),
        (0.73936463, -0.69987),
        (0.3888584, -0.928921),
        (-0.128635, -0.94627),
        (-0.5342567, -0.856044),
        (-0.4852847, -0.7544567),
        (0.1528741, -0.779696),
        (0.677304, -0.52244),
        (0.69770106, 0.097088),
        (0.4147023, 0.5611049),
        (-0.2481776, 0.556),
        (-0.571969, 0.31379975),
        (-0.58978244, -0.120328),
        (-0.378090609, -0.33202011),
        (-0.1003044, -0.420467),
        (0.15974848, -0.303188),
        (0.233685, -0.09922),
        (0.215838, 0.066494),
        (0.1011091, 0.15572837),
        (-0.0697099, 0.16082745),
        (-0.20483549, 0.122584),
        (-0.29406933, 0.0231523),
        (-0.27622569, -0.12981991),
        (-0.15129519, -0.19355837),
        (-0.06206135, -0.1910088),
        (-0.0238182, -0.11452268)
    ]
        out_shape = Polygon(*args, vertices=points)
    elif shape_type == "seaurchin":
        points = [
            (0, 1),
            (0.01913417, 0.046194),
            (0.8, 0.8),
            (0.046194, 0.01913417),
            (1, 0),
            (0.046194, -0.01913417),
            (0.8, -0.8),
            (0.01913417, -0.046194),
            (0, -1),
            (-0.01913417, -0.046194),
            (-0.8, -0.8),
            (-0.046194, -0.01913417),
            (-1, 0),
            (-0.046194, 0.01913417),
            (-0.8, 0.8),
            (-0.01913417, 0.046194)
        ]
        out_shape = Polygon(*args, vertices=points)
        
    elif shape_type == "peace":
        args[0] //= 2
        
        cir = Circle(*args, r=1.0, center=(0, 0), outward_normal=True)
        
        args[0] //= 4
        
        right_poly = Polygon(*args, vertices=[
            (0.1225, 0.8),
            (0.46, 0.64),
            (0.7, 0.4),
            (0.8, 0.0),
            (0.7, -0.4),
            (0.1225, 0.0)
        ], outward_normal=False)
        left_poly = Polygon(*args, vertices=[
            (-0.1225, 0.0),
            (-0.7, -0.4),
            (-0.8, 0.0),
            (-0.7, 0.4),
            (-0.46, 0.64),
            (-0.1225, 0.8)
        ], outward_normal=False)
        
        right_tri = Polygon(*args, vertices=[
            (0.1225, -0.3),
            (0.55, -0.6),
            (0.1225, -0.8)
        ], outward_normal=False)
        left_tri = Polygon(*args, vertices=[
            (-0.1225, -0.3),
            (-0.1225, -0.8),
            (-0.55, -0.6)
        ], outward_normal=False)
        
        out_shape = Union(shapes=[cir, right_poly, left_poly, right_tri, left_tri])
        
    elif shape_type == "boomerangs":
        args[0] //= 2
        b1 = Polygon(*args, vertices=[
            (0.6, -0.6),
            (0.8, 0.6),
            (0.4, -0.2),
            (-0.4, -0.2)
        ], outward_normal=False)
        b2 = Polygon(*args, vertices=[
            (-0.6, 0.6),
            (-0.8, -0.6),
            (-0.4, 0.2),
            (0.4, 0.2)
        ], outward_normal=False)
        out_shape = Union(shapes=[b1, b2])
        
    elif shape_type == "fragments":
        args[0] //= 15
        f = []
        f.append(get_tri_points((1, 0), 0.2))
        f.append(get_tri_points((0.25, 0.433), 0.2))
        f.append(get_tri_points((-0.5, 0.866), 0.2))
        f.append(get_tri_points((-0.5, 0), 0.2))
        f.append(get_tri_points((0.625, -0.2165), 0.2))
        f.append(get_tri_points((0.25, -0.433), 0.2))
        f.append(get_tri_points((-0.125, -0.6495), 0.2))
        f.append(get_tri_points((-0.125, 0.6495), 0.2))
        f.append(get_tri_points((-0.5, -0.866), 0.2))
        f.append(get_tri_points((0.25, 0.0), 0.2))
        f.append(get_tri_points((-0.125, -0.2165), 0.2))
        f.append(get_tri_points((0.625, 0.2165), 0.2))
        f.append(get_tri_points((-0.5, -0.433), 0.2))
        f.append(get_tri_points((-0.5, 0.433), 0.2))
        f.append(get_tri_points((-0.125, 0.2165), 0.2))
        
        f = [Polygon(*args, vertices=points, outward_normal=True) for points in f]
        
        out_shape = Union(shapes=f)
    elif shape_type == "house":
        
        args[0] //= 4
        facade = Polygon(*args, vertices=[
            (-1, 0.18),
            (1, 0.18),
            (1, -1),
            (-1, -1)
        ])
        roof = Polygon(*args, vertices=[
            (-0.5, 1),
            (1, 0.35),
            (-1, 0.35)
        ])
        
        args[0] //= 2
        window = Polygon(*args, vertices=[
            (-0.7, 0.0),
            (0.2, 0.0),
            (0.2, -0.4),
            (-0.7, -0.4)
        ], outward_normal=False)
        door = Polygon(*args, vertices=[
            (0.55, -0.4),
            (0.8, -0.4),
            (0.8, -0.85),
            (0.55, -0.85)
        ], outward_normal=False)
        handle = Circle(*args, r=0.05, center=(0.75, -0.65), outward_normal=True)
        chimney = Polygon(*args, vertices=[
            (0.4, 1),
            (0.6, 1),
            (0.6, 0.6),
            (0.4, 0.7)
        ])
        out_shape = Union(shapes=[facade, window, door, handle, roof, chimney])
    else:
        raise Warning("Unsupportaed shape")

    return out_shape
