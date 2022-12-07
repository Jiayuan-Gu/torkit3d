"""PointNet for classification.

See also:
    https://github.com/charlesq34/pointnet

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

from torkit3d.nn import mlp
from torkit3d.nn.modules.linear import LinearNormAct

__all__ = ["PointNet", "TNet"]


class TNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=None,
        local_channels=(64, 128, 1024),
        global_channels=(512, 256),
    ):
        super().__init__()

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.mlp_local = mlp(in_channels, local_channels, ndim=1)
        self.mlp_global = mlp(local_channels[-1], global_channels)
        self.linear = nn.Linear(
            global_channels[-1], in_channels * out_channels, bias=True
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        # x: [B, C, N]
        x = self.mlp_local(x)  # [B, D1, N]
        x, _ = torch.max(x, 2)  # [B, D1]
        x = self.mlp_global(x)  # [B, D2]
        x = self.linear(x)  # [B, C' * C]
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(
            self.out_channels, self.in_channels, dtype=x.dtype, device=x.device
        )
        x = x.add(I)  # broadcast add, [B, C', C]
        return x

    def reset_parameters(self):
        # Initialize linear transform to 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class PointNet(nn.Module):
    def __init__(
        self,
        in_channels=0,
        hidden_sizes=(64, 64, 64, 128, 1024),
        normalization="bn",
        no_first_normalization=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = hidden_sizes[-1]
        self.mlp = mlp(
            3 + in_channels, hidden_sizes, ndim=1, normalization=normalization
        )

        if no_first_normalization:
            # Remove normalization in the first layer
            self.mlp[0] = LinearNormAct(3 + in_channels, hidden_sizes[0], ndim=1)

    def forward(
        self,
        points: torch.Tensor,  # [B, 3, N]
        feats: torch.Tensor = None,  # [B, C, N]
        mask: torch.Tensor = None,  # [B, N]
        **kwargs
    ) -> dict:
        # Sanity check
        assert points.dim() == 3 and points.size(1) == 3, points.size()

        x = points if feats is None else torch.cat([points, feats], dim=1)
        x = self.mlp(x)
        if mask is not None:
            x = torch.where(mask.unsqueeze(1), x, torch.zeros_like(x))
        x, max_inds = torch.max(x, 2)
        return {"feats": x, "max_inds": max_inds}


def test():
    bs = 32
    n = 1024
    d = 4

    points = torch.randn(bs, 3, n)
    feats = torch.randn(bs, d, n)
    mask = torch.rand(bs, n) > 0.5

    # Basic
    model = PointNet(0)
    print(model)
    outs = model(points)
    for k, v in outs.items():
        print(k, v.shape)

    # With features
    model = PointNet(d)
    print(model)
    outs = model(points, feats)
    for k, v in outs.items():
        print(k, v.shape)

    # With masks
    outs = model(points, feats, mask)
    for k, v in outs.items():
        print(k, v.shape)


if __name__ == "__main__":
    test()
