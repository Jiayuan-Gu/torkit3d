"""PointNet++"""

import torch.nn as nn
from torkit.nn.functional import batch_index_select
from torkit3d.layers.pointnet2 import SetAbstraction, BallQuery, FeaturePropagation
from torkit3d.ops.farthest_point_sample import farthest_point_sample

__all__ = ["PN2SSG"]


class PN2SSG(nn.Module):
    """PointNet++ single-scale-grouping.

    The default parameters are for ScanNet.
    """

    def __init__(
        self,
        in_channels=0,
        num_samples=(2048, 512, 128, 32),
        radius_list=(0.1, 0.2, 0.4, 0.8),
        max_neighbors=32,
        use_xyz=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 128 + in_channels
        self.num_samples = num_samples
        self.use_xyz = use_xyz

        # sanity check
        assert len(num_samples) == len(radius_list) == 4

        # Set abstraction layers
        sa_layers = [
            SetAbstraction(
                in_channels=in_channels,
                mlp_channels=(32, 32, 64),
                query_neighbor=BallQuery(
                    radius=radius_list[0], max_neighbors=max_neighbors
                ),
                use_xyz=use_xyz,
            ),
            SetAbstraction(
                in_channels=64,
                mlp_channels=(64, 64, 128),
                query_neighbor=BallQuery(
                    radius=radius_list[1], max_neighbors=max_neighbors
                ),
                use_xyz=use_xyz,
            ),
            SetAbstraction(
                in_channels=128,
                mlp_channels=(128, 128, 256),
                query_neighbor=BallQuery(
                    radius=radius_list[2], max_neighbors=max_neighbors
                ),
                use_xyz=use_xyz,
            ),
            SetAbstraction(
                in_channels=256,
                mlp_channels=(256, 256, 512),
                query_neighbor=BallQuery(
                    radius=radius_list[3], max_neighbors=max_neighbors
                ),
                use_xyz=use_xyz,
            ),
        ]
        self.sa_layers = nn.ModuleList(sa_layers)

        # Feature propagation layers
        fp_layers = [
            FeaturePropagation(in_channels=512 + 256, mlp_channels=(256, 256)),
            FeaturePropagation(in_channels=256 + 128, mlp_channels=(256, 256)),
            FeaturePropagation(in_channels=256 + 64, mlp_channels=(256, 128)),
            FeaturePropagation(
                in_channels=128 + in_channels, mlp_channels=(128, 128, 128)
            ),
        ]
        self.fp_layers = nn.ModuleList(fp_layers)

        # Initialize
        self.reset_parameters()

    def forward(self, points, points_feature=None):
        # points: [B, 3, N], points_feature: [B, C, N]
        fps_index = farthest_point_sample(points, self.num_samples[0], True)  # [B, N1]
        xyz_fps = batch_index_select(points, fps_index, dim=2)  # [B, 3, N1]

        # Set abstraction layers
        xyz0, feature0 = points, points_feature
        xyz1 = xyz_fps[..., 0 : self.num_samples[0]]
        feature1 = self.sa_layers[0](xyz0, xyz1, feature0)
        xyz2 = xyz_fps[..., 0 : self.num_samples[1]]
        feature2 = self.sa_layers[1](xyz1, xyz2, feature1)
        xyz3 = xyz_fps[..., 0 : self.num_samples[2]]
        feature3 = self.sa_layers[2](xyz2, xyz3, feature2)
        xyz4 = xyz_fps[..., 0 : self.num_samples[3]]
        feature4 = self.sa_layers[3](xyz3, xyz4, feature3)

        # Feature propagation layers
        fp_feature3 = self.fp_layers[0](xyz4, feature4, xyz3, feature3)
        fp_feature2 = self.fp_layers[1](xyz3, fp_feature3, xyz2, feature2)
        fp_feature1 = self.fp_layers[2](xyz2, fp_feature2, xyz1, feature1)
        fp_feature0 = self.fp_layers[3](xyz1, fp_feature1, xyz0, feature0)

        pred_dict = {"feature": fp_feature0}
        pred_dict.update(
            fp_feature0=fp_feature0,
            fp_feature1=fp_feature1,
            fp_feature2=fp_feature2,
            fp_feature3=fp_feature3,
        )

        return pred_dict

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def main():
    import torch

    data_batch = dict()
    data_batch["points"] = torch.randn(5, 3, 3000)
    data_batch["points_feature"] = torch.randn(5, 4, 3000)
    data_batch = {k: v.cuda() for k, v in data_batch.items()}

    # net = PN2SSG(0)
    net = PN2SSG(0 + 4)
    net = net.cuda()
    print(net)
    pred_dict = net(**data_batch)
    for k, v in pred_dict.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
