import torch
from torkit.nn import mlp2d_bn_relu
from torkit3d.ops.group_points import group_points


class EdgeConvBlock(torch.nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = conv_channels[-1]

        self.convs = mlp2d_bn_relu(in_channels * 2, conv_channels)

    def forward(self, query_feature, key_feature, key_ind):
        # query_feature: [B, C, N1], key_feature: [B, C, N2], key_ind: [B, N1, K]
        edge_feature = get_edge_feature(query_feature, key_feature, key_ind)  # [B, 2C, N1, K]
        edge_feature = self.convs(edge_feature)
        output, _ = edge_feature.max(dim=3)
        return output


def get_edge_feature_builtin(query_feature, key_feature, key_ind, include_query=True) -> torch.Tensor:
    b, c, n1 = query_feature.size()
    _, _, n2 = key_feature.size()
    _, _, k = key_ind.size()
    assert query_feature.size(0) == key_feature.size(0) == key_ind.size(0), \
        [query_feature.size(0), key_feature.size(0), key_ind.size(0)]
    assert key_feature.size(1) == c
    assert key_ind.size(1) == n1

    query_feature_expand = query_feature.unsqueeze(3).expand(b, c, n1, k)
    key_feature_expand = key_feature.unsqueeze(2).expand(b, c, n1, n2)
    key_ind_expand = key_ind.unsqueeze(1).expand(b, c, n1, k)
    key_feature_knn = torch.gather(key_feature_expand, 3, key_ind_expand)  # [B, C, N1, K]

    edge_feature = key_feature_knn - query_feature_expand
    if include_query:
        edge_feature = torch.cat([edge_feature, query_feature_expand], dim=1)
    return edge_feature


def get_edge_feature_custom(query_feature, key_feature, key_ind, include_query=True) -> torch.Tensor:
    b, c, n1 = query_feature.size()
    _, _, n2 = key_feature.size()
    _, _, k = key_ind.size()
    assert query_feature.size(0) == key_feature.size(0) == key_ind.size(0), \
        [query_feature.size(0), key_feature.size(0), key_ind.size(0)]
    assert key_feature.size(1) == c
    assert key_ind.size(1) == n1

    query_feature_expand = query_feature.unsqueeze(3).expand(b, c, n1, k)
    key_feature_knn = group_points(key_feature, key_ind)

    edge_feature = key_feature_knn - query_feature_expand
    if include_query:
        edge_feature = torch.cat([edge_feature, query_feature_expand], dim=1)
    return edge_feature


# get_edge_feature = get_edge_feature_builtin
get_edge_feature = get_edge_feature_custom
