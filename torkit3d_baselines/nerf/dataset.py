import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from torch.utils.data import Dataset


def get_rays_at_camera(H, W, K, convention="blender"):
    v, u = np.indices([H, W])  # [H, W] each
    # Assume no skew
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    if convention in ["blender", "opengl"]:
        # Blender/OpenGL convension
        rays_d = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], axis=-1)
    else:
        raise NotImplementedError(convention)
    return rays_d


def get_rays_at_world(H, W, K, c2w, convention="blender"):
    rays_d = get_rays_at_camera(H, W, K, convention=convention)  # [H, W, 3]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_d = rays_d @ c2w[:3, :3].T  # [H, W, 3]
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # [H, W, 3]
    return rays_o, rays_d


class NeRFSyntheticDataset(Dataset):
    def __init__(self, root_dir, split, image_size=None, n_rays=1024) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.n_rays = n_rays

        with open(self.root_dir / f"transforms_{split}.json", "r") as f:
            self.meta = json.load(f)

        self.cache_images = {}

    def __len__(self):
        return len(self.meta["frames"])

    def __getitem__(self, index):
        frame = self.meta["frames"][index]

        if index in self.cache_images:
            image = self.cache_images[index]
        else:
            image_path = self.root_dir / (frame["file_path"] + ".png")
            image = np.asarray(imageio.imread(image_path))
            image = image[..., :3]

        # Blender convention
        c2w = np.float32(frame["transform_matrix"])

        # Compute intrinsic according to image size
        H, W = image.shape[:2]
        camera_angle_x = float(self.meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(camera_angle_x / 2)
        K = np.float32([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

        if self.image_size is not None:
            H_dst, W_dst = self.image_size
            if (W_dst, H_dst) != (W, H):
                image = cv2.resize(image, (W_dst, H_dst), interpolation=cv2.INTER_AREA)
                K[0] *= W_dst / W
                K[1] *= H_dst / H

        # Normalize image
        if image.dtype == np.uint8:
            image = image / 255.0
        image = image.astype(np.float32, copy=False)

        # Cache image
        if index not in self.cache_images:
            self.cache_images[index] = image

        # return dict(image=image, pose=c2w, intrinsic=K)

        # Get rays
        H, W = image.shape[:2]
        rays_o, rays_d = get_rays_at_world(H, W, K, c2w, convention="blender")
        rays = np.concatenate([rays_o, rays_d], axis=-1, dtype=np.float32)
        rays = rays.reshape(-1, rays.shape[-1])
        rays_rgb = image.reshape(-1, 3)

        # Sample rays
        if self.n_rays is not None:
            inds = np.random.choice(len(rays), self.n_rays, replace=False)
            rays = rays[inds]
            rays_rgb = rays_rgb.reshape(-1, 3)[inds]

        return dict(rays=rays, rays_rgb=rays_rgb)


def test():
    root_dir = Path(__file__).parent / "data/nerf_synthetic/lego"
    dataset = NeRFSyntheticDataset(root_dir, "train", image_size=(200, 200))
    for i in range(len(dataset)):
        data = dataset[i]
        # cv2.imshow("nerf_dataset", data["image"][..., :3][..., ::-1])
        # cv2.waitKey(0)
        for k, v in data.items():
            print(k, v.shape)
        # print(data["pose"])
        # print(data["intrinsic"])

        # -------------------------------------------------------------------------- #
        # Visualize rays
        # -------------------------------------------------------------------------- #
        # # isort: off
        # import matplotlib.pyplot as plt
        # from pytransform3d.plot_utils import plot_vector, make_3d_axis

        # ax = make_3d_axis(ax_s=5)
        # for ray in data["rays"][:10]:
        #     plot_vector(
        #         ax=ax,
        #         # A vector is defined by start, direction, and s (scaling)
        #         start=ray[0:3],
        #         direction=ray[3:6],
        #         s=4,
        #         color="orange",
        #     )
        # plt.show()
        # plt.close()
