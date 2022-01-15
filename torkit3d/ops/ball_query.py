import torch

from torkit3d import _C


class BallQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, radius, max_neighbors):
        index = _C.ball_query_cuda(query, key, radius, max_neighbors)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def ball_query(
    query: torch.Tensor, key: torch.Tensor, radius, max_neighbors, transpose=True
):
    """Query neighbors within a radius.

    Args:
        query (torch.Tensor): [B, 3, N1] tensor
        key (torch.Tensor): [B, 3, N2] tensor
        radius (float): radius to query
        max_neighbors (int): maximum neighbors to extract. If not enough, padded with the first one.
        transpose (bool): whether to transpose inputs to [B, N, C] format.

    Returns:
        torch.Tensor: [B, N2, K] integer tensor.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    index = BallQueryFunction.apply(query, key, radius, max_neighbors)
    return index
