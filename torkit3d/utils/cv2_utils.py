import numpy as np
import cv2
import warnings

VERTEX_COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)

        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(image: np.ndarray,
                         center, size, rotation,
                         extrinsic, intrinsic,
                         color=(0, 255, 0), thickness=1):
    """Draw a projected 3D bounding box on the image.

    Args:
        image: [H, W, 3]
        center: [3]
        size: [3]
        rotation: [3, 3]
        extrinsic: [4, 4]
        intrinsic: [3, 3]
        color: [3]
        thickness: thickness of lines

    Returns:
        np.ndarray: image with box drawn (in-place)
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)
    z = corners_image[:, 2]

    for (i, j) in edges:
        if z[i] <= 0.0 or z[j] <= 0.0:
            warnings.warn('Some corners are behind the camera.')
            continue
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image


def test_draw_projected_box3d():
    from scipy.spatial.transform import Rotation

    image = np.zeros([256, 256, 3])
    image[:128, :128] = 1.0
    # world: -x forward, y right, z up
    # cam: z forward, x right, y down
    extrinsic = np.array([[0., 1., 0., 0.],
                          [0., 0., -1., 0.],
                          [-1., 0., 0., 2.],
                          [0., 0., 0., 1.]])
    intrinsic = np.array([[128, 0, 128], [0, 128, 128], [0, 0, 1]])
    angle = 0
    # angle = 10
    rotation = Rotation.from_euler('x', angle, degrees=True).as_matrix()
    draw_projected_box3d(image,
                         center=[0., 0., 0.], size=[1., 1., 1.], rotation=rotation,
                         extrinsic=extrinsic, intrinsic=intrinsic,
                         )
    cv2.imshow('debug', image)
    cv2.waitKey(0)
