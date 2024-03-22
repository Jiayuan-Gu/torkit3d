import torch

from torkit3d import _C


class SampleFarthestPointsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_samples):
        index = _C.sample_farthest_points_cuda(points, num_samples)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def sample_farthest_points(points: torch.Tensor, num_samples, transpose=False):
    """Sample farthest points.

    Args:
        points (torch.Tensor): [B, N, 3] or [B, 3, N] if @transpose is True.
        num_samples (int): the number of points to sample
        transpose (bool): whether to transpose points.

    Returns:
        torch.Tensor: [B, N], sampled indices.
    """
    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    return SampleFarthestPointsFunction.apply(points, num_samples)
