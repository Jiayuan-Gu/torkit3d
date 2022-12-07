import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------- #
# Depth
# ---------------------------------------------------------------------------- #
def unproject_depth(depth_image: np.ndarray, intrinsic: np.ndarray, offset):
    """Unproject a depth image to a 3D poin cloud.

    Args:
        depth_image: [H, W]
        intrinsic: [3, 3]
        offset: offset of x and y indices.

    Returns:
        points: [H, W, 3]
    """
    v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
    z = depth_image  # [H, W]
    uv1 = np.stack([u + offset, v + offset, np.ones_like(z)], axis=-1)
    points = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]
    return points


def get_intrinsic_matrix(width, height, fov, degree=False):
    """Get the camera intrinsic matrix according to image size and fov."""
    if degree:
        fov = np.deg2rad(fov)
    f = (width / 2.0) / np.tan(fov / 2.0)
    xc = (width - 1.0) / 2.0
    yc = (height - 1.0) / 2.0
    K = np.array([[f, 0, xc], [0, f, yc], [0, 0, 1.0]])
    return K


# ---------------------------------------------------------------------------- #
# Point cloud
# ---------------------------------------------------------------------------- #
def pad_with_first_or_clip(array: np.array, n: int):
    """Pad or clip an array with the first item.
    It is usually used for sampling a fixed number of points (PointNet and variants).
    """
    if array.shape[0] >= n:
        return array[:n]
    else:
        pad = np.repeat(array[0:1], n - array.shape[0], axis=0)
        return np.concatenate([array, pad], axis=0)


def normalize_points(points):
    """Centralize point clouds and scale them by l2-norm."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    centroid = np.mean(points, axis=0)  # [N]
    points = points - centroid  # [N, 3]
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    return points / norm


def rotate_points(points, axis, angle):
    R = Rotation.from_rotvec(np.array(axis) * angle).as_matrix()
    return points @ R.T
