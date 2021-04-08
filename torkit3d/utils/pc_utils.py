import numpy as np


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


def pad_with_first_or_clip(array: np.array, n: int):
    """Pad or clip an array with the first item.
    It is usually used for sampling a fixed number of points (PointNet and variants).
    """
    if array.shape[0] >= n:
        return array[:n]
    else:
        pad = np.repeat(array[0:1], n - array.shape[0], axis=0)
        return np.concatenate([array, pad], axis=0)
