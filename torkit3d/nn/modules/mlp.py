import torch
from torch import nn

from .linear import LinearNormAct

__all__ = ["mlp"]


def mlp(c_in, hidden_sizes, ndim=0, normalization="bn", dropout_p=0.0):
    layers = []
    for c_out in hidden_sizes:
        layer = LinearNormAct(
            c_in, c_out, ndim=ndim, normalization=normalization, dropout_p=dropout_p
        )
        layers.append(layer)
        c_in = c_out
    return nn.Sequential(*layers)
