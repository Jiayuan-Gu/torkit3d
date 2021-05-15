"""PointNet++ modules."""

import torch
import torch.nn as nn

from torkit.nn import mlp1d_bn_relu, mlp2d_bn_relu
from torkit3d.ops.group_points import group_points
from torkit3d.ops.ball_query import ball_query
from torkit3d.ops.knn_distance import knn_distance
from torkit3d.ops.interpolate_feature import interpolate_feature


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction module."""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 query_neighbor,
                 use_xyz):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.query_neighbor = query_neighbor
        self.use_xyz = use_xyz

        self.mlp = mlp2d_bn_relu(in_channels + (3 if use_xyz else 0), mlp_channels)

    def forward(self, in_xyz: torch.Tensor, out_xyz: torch.Tensor, in_feature: torch.Tensor):
        """

        Args:
            in_xyz: [B, 3, N1]
            out_xyz: [B, 3, N2]
            in_feature: [B, C1, N1]

        Returns:
            torch.Tensor: [B, C2, N2], output features
        """
        # Sanity check
        assert in_xyz.size(0) == out_xyz.size(0)
        assert in_xyz.size(1) == out_xyz.size(1) == 3

        with torch.no_grad():
            # [B, N2, K]
            nbr_index = self.query_neighbor(out_xyz, in_xyz)

        if in_feature is not None:
            nbr_feature = group_points(in_feature, nbr_index)
        else:
            nbr_feature = in_xyz.new_empty([in_xyz.size(0), 0, nbr_index.size(1), nbr_index.size(2)])

        if self.use_xyz:
            # centralize
            nbr_xyz = group_points(in_xyz, nbr_index)  # [B, 3, N2, K]
            nbr_xyz = nbr_xyz - out_xyz.unsqueeze(-1)
            nbr_feature = torch.cat([nbr_feature, nbr_xyz], dim=1)

        out_feature = self.mlp(nbr_feature)
        out_feature, _ = torch.max(out_feature, dim=3)
        return out_feature

    def extra_repr(self):
        attributes = ['use_xyz']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])


class BallQuery(torch.nn.Module):
    def __init__(self, radius, max_neighbors):
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors

    def forward(self, query, key):
        nbr_index = ball_query(query, key, self.radius, self.max_neighbors)
        return nbr_index

    def extra_repr(self):
        attributes = ['radius', 'max_neighbors']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])


class FeaturePropagation(nn.Module):
    """PointNet++ feature propagation module"""

    def __init__(self, in_channels, mlp_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        self.mlp = mlp1d_bn_relu(in_channels, mlp_channels)
        self.interpolator = FeatureInterpolator()

    def forward(self, in_xyz, in_feature, out_xyz, out_feature=None):
        interp_feature = self.interpolator(out_xyz, in_xyz, in_feature)
        if out_feature is not None:
            interp_feature = torch.cat([interp_feature, out_feature], dim=1)
        fp_feature = self.mlp(interp_feature)
        return fp_feature


class FeatureInterpolator(nn.Module):
    def __init__(self, num_neighbors=3, eps=1e-10):
        super().__init__()
        self.num_neighbors = num_neighbors
        self._eps = eps

    def forward(self, query_xyz, key_xyz, key_feature):
        """Interpolate features from key to query.

        Args:
            query_xyz: [B, 3, N1]
            key_xyz: [B, 3, N2]
            key_feature: [B, C2, N1]

        Returns:
            torch.Tensor: [B, C2, N1], propagated feature
        """
        with torch.no_grad():
            # index: [B, N1, K], distance: [B, N1, K]
            index, distance = knn_distance(query_xyz, key_xyz, self.num_neighbors)
            inv_distance = 1.0 / torch.clamp(distance, min=self._eps)
            norm = torch.sum(inv_distance, dim=2, keepdim=True)
            weight = inv_distance / norm

        interp_feature = interpolate_feature(key_feature, index, weight)
        return interp_feature

    def extra_repr(self):
        attributes = ['num_neighbors']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])
