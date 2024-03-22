import torch

from torkit3d import _C


class KNNDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_xyz, key_xyz, k, version):
        return _C.knn_distance_cuda(query_xyz, key_xyz, k, version)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    sqrt_distance: bool = False,
    transpose: bool = False,
    version: int = 1,
):
    """Find k nearest neighbors along with distances in the key set for each point in the query set.

    Args:
        query: [B, N1, 3], query points. [B, 3, N1] if @transpose is True.
        key: [B, N2, 3], key points. [B, 3, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        sqrt_distance: whether to sqrt the distance
        transpose: whether to transpose the last two dimensions.
        version: different implementations of cuda kernel.

    Returns:
        index: [B, N1, K], indices of the k nearest neighbors in the key.
        distance: [B, N1, K], distances to the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    distance, index = KNNDistanceFunction.apply(query, key, k, version)
    if sorted:
        distance, sort_idx = torch.sort(distance, dim=2)
        index = torch.gather(index, 2, sort_idx)
    if sqrt_distance:
        distance = torch.sqrt(distance)
    return distance, index
