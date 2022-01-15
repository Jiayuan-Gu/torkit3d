import torch


def compute_knn_builtin(
    query: torch.Tensor, key: torch.Tensor, k: int, exclude_first=False
):
    """Compute k nearest neighbors with builtin operators.

    Args:
        query: [B, D, N1]
        key:  [B, D, N2]
        k: k-nn
        exclude_first: whether to exclude the first neighbor
            (used to remove self when key=query)

    Returns:
        torch.Tensor: [B, N1, K]
    """
    distance = torch.cdist(query.transpose(1, 2), key.transpose(1, 2))  # [B, N1, N2]
    k_actual = k + (1 if exclude_first else 0)
    _, knn_ind = torch.topk(distance, k_actual)
    return knn_ind[..., (1 if exclude_first else 0) :]
