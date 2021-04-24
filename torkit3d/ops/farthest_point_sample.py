import torch
from torkit3d import _C


class FarthestPointSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_samples):
        index = _C.farthest_point_sample_cuda(points, num_samples)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def farthest_point_sample(points: torch.Tensor, num_samples, transpose=True):
    """Farthest point sample.

    Args:
        points (torch.Tensor): [B, 3, N]
        num_samples (int): the number of points to sample
        transpose (bool): whether to transpose points.
            If false, then points should be [B, N, 3].

    Returns:
        torch.Tensor: [B, N], sampled indices.
    """
    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    return FarthestPointSampleFunction.apply(points, num_samples)
