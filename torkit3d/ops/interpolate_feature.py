import torch
from torkit3d import _C


class InterpolateFeature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, index, weight):
        b, c, n1 = feature.size()
        ctx.save_for_backward(index, weight)
        ctx.n1 = n1
        interpolated_feature = _C.interpolate_forward_cuda(
            feature.contiguous(),
            index.contiguous(),
            weight.contiguous())
        return interpolated_feature

    @staticmethod
    def backward(ctx, *grad_out):
        index, weight = ctx.saved_tensors
        n1 = ctx.n1
        grad_input = _C.interpolate_backward_cuda(
            grad_out[0].contiguous(),
            index.contiguous(),
            weight.contiguous(),
            n1)
        return grad_input, None, None


def interpolate_feature(feature: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """Interpolate features given indices and weights.

    Args:
       feature (torch.Tensor): [B, C, N1], features of key points
       index (torch.Tensor): [B, N2, K], indices of key points to interpolate
       weight (torch.Tensor): [B, N2, K], weights to interpolate

    Returns:
       interpolated_feature: (B, C, N2)
    """
    return InterpolateFeature.apply(feature, index, weight)
