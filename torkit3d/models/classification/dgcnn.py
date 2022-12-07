import torch
import torch.nn as nn

from torkit3d.nn import mlp1d_bn_relu, mlp_bn_relu
from torkit3d.ops.edge_conv import EdgeConvBlock
from torkit3d.ops.knn import compute_knn_builtin

__all__ = ["DGCNN"]


class TNet(nn.Module):
    """Transformation Network for DGCNN

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        conv_channels (tuple of int): the numbers of channels of edge convolution layers
        local_channels (tuple of int): the numbers of channels in local mlp
        global_channels (tuple of int): the numbers of channels in global mlp
        k: the number of neareast neighbours for edge feature extractor

    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        conv_channels=(64, 128),
        local_channels=(1024,),
        global_channels=(512, 256),
        k=20,
    ):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.edge_conv = EdgeConvBlock(in_channels, conv_channels)
        self.mlp_local = mlp1d_bn_relu(conv_channels[-1], local_channels)
        self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
        self.linear = nn.Linear(
            global_channels[-1], self.in_channels * self.out_channels, bias=True
        )

        self.reset_parameters()

    def forward(self, x):
        # input x: (batch_size, in_channels, num_points)
        knn_ind = compute_knn_builtin(x, x, self.k)
        x = self.edge_conv(x, x, knn_ind)
        x = self.mlp_local(x)  # (batch_size, local_channels[-1], num_points)
        x, _ = torch.max(x, 2)
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, device=x.device)
        x = x.add(I)  # broadcast first dimension
        return x

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear.weight)


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
        feature_knn=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = local_channels[-1]
        self.k = k
        self.feature_knn = feature_knn

        self.transform_input = TNet(in_channels, in_channels, k=k)

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

    def forward(self, points, points_feature=None, knn_ind=None, **kwargs):
        # points: [B, 3, N]; points_feature: [B, C, N]
        features = []
        if points_feature is not None:
            x = torch.cat([points, points_feature], dim=1)
        else:
            x = points

        trans_input = self.transform_input(x)
        x = torch.bmm(trans_input, x)

        for edge_conv in self.edge_convs:
            knn_ind = compute_knn_builtin(x, x, self.k)
            x = edge_conv(x, x, knn_ind)
            features.append(x)

        concat_feature = torch.cat(features, dim=1)
        local_feature = self.mlp_local(concat_feature)
        global_feature, max_indices = torch.max(local_feature, 2)
        # output_feature = self.mlp_global(global_feature)

        return {"feats": global_feature, "max_indices": max_indices}

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
