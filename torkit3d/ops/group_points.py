import torch

from torkit3d import _C


class GroupPointsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        ctx.save_for_backward(index)
        ctx.num_points = points.size(2)
        return _C.group_points_forward_cuda(points.contiguous(), index.contiguous())

    @staticmethod
    def backward(ctx, *grad_output):
        index = ctx.saved_tensors[0]
        grad_input = _C.group_points_backward_cuda(
            grad_output[0].contiguous(), index.contiguous(), ctx.num_points
        )
        return grad_input, None


def group_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: [B, C, N1]
        index: [B, N2, K], indices of neighbors.

    Returns:
        torch.Tensor: [B, C, N2, K], grouped points.
    """
    return GroupPointsFunction.apply(points, index)
