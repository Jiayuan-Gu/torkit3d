"""PointNet for segmentation.

See also:
    https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py

References:
    @article{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
      author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1612.00593},
      year={2016}
    }
"""

import torch
import torch.nn as nn

from torkit3d.models.classification.pointnet import TNet
from torkit3d.nn import mlp1d_bn_relu

__all__ = ["PointNetPartSeg"]


class PointNetPartSeg(nn.Module):
    def __init__(
        self,
        in_channels=0,
        hidden_sizes=(64, 128, 128, 512, 2048),
        use_input_transform: bool = True,
        use_feature_transform: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_sizes = hidden_sizes

        # Input transform (xyz only)
        self.use_input_transform = use_input_transform
        if self.use_input_transform:
            self.transform_input = TNet(3)

        # Feature transform
        self.use_feature_transform = use_feature_transform
        if use_feature_transform is not None:
            dim = hidden_sizes[use_feature_transform]
            self.trans_dim = dim
            self.transform_feature = TNet(dim)

        self.mlp = mlp1d_bn_relu(3 + in_channels, hidden_sizes)

        self.reset_parameters()

    @property
    def out_channels(self):
        return sum(self.hidden_sizes) + self.hidden_sizes[-1] + self.trans_dim

    def forward(
        self,
        points: torch.Tensor,  # [B, 3, N]
        feats: torch.Tensor = None,  # [B, C, N]
        mask: torch.Tensor = None,  # [B, N]
        **kwargs
    ) -> dict:
        # Sanity check
        assert points.dim() == 3 and points.size(1) == 3, points.size()
        outs = dict(feats=[])

        # Input transform
        if self.use_input_transform:
            T = self.transform_input(points)
            outs["T_points"] = T
            points = torch.bmm(T, points)

        x = points if feats is None else torch.cat([points, feats], 1)

        for i, layer in enumerate(self.mlp):
            x = layer(x)
            outs["feats"].append(x)
            if i == self.use_feature_transform:
                T = self.transform_feature(x)
                outs["T_feats"] = T
                x = torch.bmm(T, x)
                outs["feats"].append(x)

        if mask is not None:
            x = torch.where(mask.unsqueeze(1), x, torch.zeros_like(x))
        global_feats, max_inds = torch.max(x, 2)
        outs["feats"].append(global_feats.unsqueeze(-1).expand_as(x))

        return outs

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # NOTE(jigu): The momentum of BN is constant,
            # instead of being decayed as in the original implementation.
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = 0.01


def test():
    bs = 32
    n = 1024
    d = 4

    points = torch.randn(bs, 3, n)
    feats = torch.randn(bs, d, n)
    mask = torch.rand(bs, n) > 0.5

    def print_outs(outs):
        for k, v in outs.items():
            if isinstance(v, (tuple, list)):
                print(k, [x.shape for x in v])
            else:
                print(k, v.shape)

    # Basic
    model = PointNetPartSeg(0)
    print(model)
    outs = model(points)
    print_outs(outs)

    # With features
    model = PointNetPartSeg(d)
    print(model)
    outs = model(points, feats)
    print_outs(outs)
    # With masks
    outs = model(points, feats, mask)
    print_outs(outs)


if __name__ == "__main__":
    test()
