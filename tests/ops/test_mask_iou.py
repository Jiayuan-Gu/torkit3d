import torch
from torkit3d.ops.mask_iou import compute_mask_iou


def compute_mask_iou_naive(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    intersection = mask1.unsqueeze(1) & mask2.unsqueeze(0)
    union = mask1.unsqueeze(1) | mask2.unsqueeze(0)
    iou = intersection.sum(-1) / union.sum(-1).clamp(min=1)
    return iou


def test_compute_mask_iou():
    mask1 = torch.rand(100, 120).cuda() > 0.5
    mask2 = torch.rand(110, 120).cuda() > 0.5

    iou_naive = compute_mask_iou_naive(mask1, mask2)
    iou = compute_mask_iou(mask1, mask2)
    torch.testing.assert_close(iou, iou_naive)
