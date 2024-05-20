import torch
from torkit3d import _C


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two masks.

    Args:
        mask1: [M1, N]
        mask2: [M2, N]

    Returns:
        torch.Tensor: [M1, M2]
    """
    assert mask1.dtype == torch.bool, mask1.dtype
    assert mask2.dtype == torch.bool, mask2.dtype
    return _C.mask_iou_cuda(mask1.contiguous(), mask2.contiguous())
