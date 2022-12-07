"""PointNet++"""

import torch
import torch.nn as nn

from torkit3d.layers.pointnet2 import BallQuery, SetAbstraction
from torkit3d.nn import mlp1d_bn_relu
from torkit3d.nn.functional import batch_index_select
from torkit3d.ops.farthest_point_sample import farthest_point_sample

__all__ = ["PN2SSG"]


class PN2SSG(nn.Module):
    """PointNet++ single-scale-grouping.

    The default parameters are for ModelNet40.
    """

    def __init__(
        self,
        in_channels=0,
        sa_channels=(((64, 64, 128), (128, 128, 256))),
        num_samples=(512, 128),
        radius_list=(0.2, 0.4),
        num_neighbors=(32, 64),
        local_channels=(256, 512, 1024),
        use_xyz=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = local_channels[-1]
        self.sa_channels = sa_channels
        self.num_samples = num_samples
        self.radius_list = radius_list
        self.num_neighbors = num_neighbors
        self.use_xyz = use_xyz

        # sanity check
        assert (
            len(sa_channels)
            == len(num_samples)
            == len(radius_list)
            == len(num_neighbors)
        )

        # Set abstraction layers
        sa_layers = []
        c_in = in_channels
        for idx, c_out_list in enumerate(sa_channels):
            sa_layers.append(
                SetAbstraction(
                    in_channels=c_in,
                    mlp_channels=c_out_list,
                    query_neighbor=BallQuery(
                        radius=radius_list[idx], max_neighbors=num_neighbors[idx]
                    ),
                    use_xyz=use_xyz,
                )
            )
            c_in = c_out_list[-1]
        self.sa_layers = nn.ModuleList(sa_layers)

        # Local
        self.mlp_local = mlp1d_bn_relu(c_in, local_channels)

        # Initialize
        self.reset_parameters()

    def forward(self, points, points_feature=None, **kwargs):
        # points: [B, 3, N], points_feature: [B, C, N]
        fps_index = farthest_point_sample(points, self.num_samples[0], True)  # [B, N1]
        xyz_fps = batch_index_select(points, fps_index, dim=2)  # [B, 3, N1]

        # Set abstraction layers
        xyz = points
        feature = points_feature
        for idx, sa_layer in enumerate(self.sa_layers):
            next_xyz = xyz_fps[..., 0 : self.num_samples[idx]]
            next_feature = sa_layer(xyz, next_xyz, feature)
            xyz = next_xyz
            feature = next_feature

        # Local mlp
        local_feature = self.mlp_local(feature)
        # Max pooling
        global_feature, max_indices = torch.max(local_feature, 2)

        return {"feats": global_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def main():
    data_batch = dict()
    data_batch["points"] = torch.randn(5, 3, 3000)
    # data_batch["points_feature"] = torch.randn(5, 4, 3000)
    data_batch = {k: v.cuda() for k, v in data_batch.items()}

    net = PN2SSG(0)
    # net = PN2SSG(0 + 4)
    net = net.cuda()
    print(net)
    pred_dict = net(**data_batch)
    for k, v in pred_dict.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
