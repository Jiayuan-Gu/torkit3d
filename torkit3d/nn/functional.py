import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy loss with label smoothing

    Args:
        input (torch.Tensor): [N, C]
        target (torch.Tensor): [N]
        label_smoothing (float): smoothing factor

    Returns:
        torch.Tensor: scalar
    """
    assert input.dim() == 2 and target.dim() == 1
    assert input.size(0) == target.size(0)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (
        label_smoothing / num_classes
    )
    log_prob = F.log_softmax(input, dim=1)
    loss = (-smooth_one_hot * log_prob).sum(1).mean()
    return loss


# ---------------------------------------------------------------------------- #
# Indexing
# ---------------------------------------------------------------------------- #
def batch_index_select(input, index, dim):
    """The batched version of `torch.index_select`.

    Args:
        input (torch.Tensor): [B, ...]
        index (torch.Tensor): [B, N] or [B]
        dim (int): the dimension to index

    References:
        https://discuss.pytorch.org/t/batched-index-select/9115/7
        https://github.com/vacancy/AdvancedIndexing-PyTorch
    """

    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        assert (
            index.dim() == 2
        ), "index is expected to be 2-dim (or 1-dim), but {} received.".format(
            index.dim()
        )
        squeeze_dim = False
    assert input.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(
        input.size(0), index.size(0)
    )
    views = [1 for _ in range(input.dim())]
    views[0] = index.size(0)
    views[dim] = index.size(1)
    expand_shape = list(input.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input, dim, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out
