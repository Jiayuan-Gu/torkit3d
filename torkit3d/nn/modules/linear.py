import torch
from torch import nn

from .normalization import *

__all__ = ["LinearNormAct"]


class LinearNormAct(nn.Module):
    """Applies a linear transformation to the incoming data,
    followed by normalization and activation.
    """

    def __init__(
        self, in_channels, out_channels, ndim=0, normalization="", dropout_p=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if normalization in ["bn", "ln", "dn"]:
            bias = False
        elif normalization == "in" and ndim != 0:
            bias = False
        else:
            bias = True

        if ndim == 0:
            self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        elif ndim == 1:
            self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        elif ndim == 2:
            self.linear = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        else:
            raise NotImplementedError(ndim)

        if normalization == "bn":
            if ndim in [0, 1]:
                self.norm = nn.BatchNorm1d(out_channels)
            elif ndim == 2:
                self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == "ln":
            self.norm = LayerNorm(out_channels)
        elif normalization == "in":
            if ndim == 0:
                self.norm = None
            elif ndim == 1:
                self.norm = nn.InstanceNorm1d(out_channels, affine=True)
            elif ndim == 2:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif normalization == "":
            self.norm = None
        else:
            raise NotImplementedError(normalization)

        # TODO(jigu): support more activation
        self.act = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
