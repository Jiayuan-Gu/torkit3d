import torch
import torch.nn as nn

from torkit3d.nn import mlp1d_bn_relu, mlp_bn_relu
from torkit3d.ops.edge_conv import EdgeConvBlock
from torkit3d.ops.native import knn

__all__ = ["DGCNN"]


class DGCNN(nn.Module):
    """DGCNN for classification.

    Notes:
        1. The original implementation includes dropout for global MLPs.
    """

    def __init__(
        self,
        in_channels=3,
        edge_conv_channels=(64, 64, 64, 128),
        local_channels=(1024,),
        global_channels=(512, 256),
        k=20,
        feature_knn=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.k = k
        self.feature_knn = feature_knn

        self.edge_convs = nn.ModuleList()
        inter_channels = []
        c_in = in_channels
        for c_out in edge_conv_channels:
            if isinstance(c_out, int):
                c_out = [c_out]
            else:
                assert isinstance(c_out, (tuple, list))
            self.edge_convs.append(EdgeConvBlock(c_in, c_out))
            inter_channels.append(c_out[-1])
            c_in = c_out[-1]
        self.mlp_local = mlp1d_bn_relu(sum(inter_channels), local_channels)
        self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)

        self.reset_parameters()

    def forward(self, points, points_feature=None, knn_ind=None):
        # points: [B, 3, N]; points_feature: [B, C, N]
        features = []
        if points_feature is not None:
            x = torch.cat([points, points_feature], dim=1)
        else:
            x = points
        for edge_conv in self.edge_convs:
            if knn_ind is None:
                _, knn_ind = knn(points, points, self.k)
            elif self.feature_knn:
                _, knn_ind = knn(x, x, self.k)
            x = edge_conv(x, x, knn_ind)
            features.append(x)

        concat_feature = torch.cat(features, dim=1)
        local_feature = self.mlp_local(concat_feature)
        global_feature, max_indices = torch.max(local_feature, 2)
        output_feature = self.mlp_global(global_feature)

        return {"feature": output_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01


def main():
    points = torch.randn(5, 3, 2048).cuda()
    feature = torch.randn(5, 3, 2048).cuda()

    model = DGCNN(3).cuda()
    print(model)
    endpoints = model(points)
    for k, v in endpoints.items():
        print(k, v.shape)

    model = DGCNN(6, feature_knn=True).cuda()
    print(model)
    endpoints = model(points, feature)
    for k, v in endpoints.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
